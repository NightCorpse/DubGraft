import io
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from dubgraft.config import ProcessingConfig
from dubgraft.execution import iter_progress_times
from dubgraft.matching import MatchingResult, TimelineAnalysis, TimelineKind
from dubgraft.media import FFmpegError, MediaChapter, MediaInfo, MediaStream
from dubgraft.processing import (
    InconclusiveTimelineError,
    ProcessingError,
    _run_ffmpeg,
    _validate_output,
    _validate_reconstructed_audio,
    mux_drift_audio,
    mux_source_audio,
    process_media,
    validate_mux_compatibility,
)


def timeline(
    kind: TimelineKind,
    *,
    offset: float | None = 0,
    slope: float = 1,
    intercept: float | None = None,
) -> TimelineAnalysis:
    return TimelineAnalysis(
        kind,
        offset,
        slope,
        offset if intercept is None else intercept,
        0,
        0,
        0.8,
        1,
        (slope - 1) * 1000,
    )


def matching_result(
    kind: TimelineKind,
    *,
    offset: float | None = 0,
    slope: float = 1,
    intercept: float | None = None,
) -> MatchingResult:
    return MatchingResult(
        (),
        (),
        4,
        20,
        timeline(kind, offset=offset, slope=slope, intercept=intercept),
    )


class _FakeProcess:
    def __init__(
        self,
        output: str,
        diagnostics: io.TextIOBase,
        *,
        return_code: int = 0,
        error: str = "",
    ) -> None:
        self.stdout = io.StringIO(output)
        self.return_code = return_code
        diagnostics.write(error)

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        return self.return_code

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _RunningProcess(_FakeProcess):
    def __init__(self, output: str, diagnostics: io.TextIOBase) -> None:
        super().__init__(output, diagnostics)
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture
def processing_media(
    tmp_path: Path,
) -> tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream]:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()
    source_audio = MediaStream(
        index=3,
        kind="audio",
        codec="eac3",
        duration=999.9,
        sample_rate=48_000,
        channels=6,
        channel_layout="5.1(side)",
        language="por",
        title="Brazilian Portuguese",
    )
    target_audio = MediaStream(
        index=1,
        kind="audio",
        codec="eac3",
        language="eng",
        title="English",
        dispositions=("default",),
    )
    source_info = MediaInfo(source, "matroska,webm", 1000, None, None, (source_audio,))
    target_info = MediaInfo(
        target,
        "matroska,webm",
        1000,
        None,
        None,
        (
            MediaStream(index=0, kind="video", codec="hevc"),
            target_audio,
            MediaStream(index=2, kind="audio", codec="aac"),
            MediaStream(index=4, kind="subtitle", codec="subrip"),
        ),
        title="Target title",
        chapters=(MediaChapter(0, 500, "First"), MediaChapter(500, 1000, "Second")),
    )
    config = ProcessingConfig(source, target, tmp_path / "output.mkv")
    return config, source_info, target_info, source_audio, target_audio


def output_info(
    path: Path,
    target_info: MediaInfo,
    source_audio: MediaStream,
    *,
    language: str = "por",
    title: str = "Brazilian Portuguese",
) -> MediaInfo:
    added_audio = replace(
        source_audio,
        index=len(target_info.streams),
        language=language,
        title=title,
    )
    return MediaInfo(
        path,
        target_info.container,
        target_info.duration,
        None,
        None,
        (*target_info.streams, added_audio),
        target_info.title,
        target_info.chapters,
    )


@pytest.mark.parametrize(
    ("offset", "option", "value"),
    [(2.5, "-ss", "2.5"), (-1.25, "-itsoffset", "1.25")],
)
def test_mux_source_audio_preserves_target_and_applies_static_offset(
    offset: float,
    option: str,
    value: str,
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    source_audio = replace(source_audio, codec="flac")
    commands: list[list[str]] = []
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)
    monkeypatch.setattr(
        "dubgraft.processing._validate_output",
        lambda path, *args, **kwargs: output_info(path, target_info, source_audio),
    )
    mux_source_audio(config, target_info, source_audio, offset=offset)

    command = commands[-1]
    option_command = commands[0] if offset > 0 else command
    assert option_command[option_command.index(option) + 1] == value
    assert command[command.index("-map") + 1] == "0"
    second_map = command.index("-map", command.index("-map") + 1)
    assert command[second_map + 1] == ("1:0" if offset > 0 else "1:3")
    if offset > 0:
        assert len(commands) == 2
        assert commands[0][commands[0].index("-f") + 1] == "matroska"
        assert "trimmed-source.mka" in commands[0][-1]
    else:
        assert len(commands) == 1
    assert "-map_metadata" in command
    assert "-map_chapters" in command
    assert "-copy_unknown" in command
    assert command[command.index("-avoid_negative_ts") + 1] == "disabled"
    assert "-c" in command and command[command.index("-c") + 1] == "copy"
    assert "-metadata:s:a:2" in command
    assert "language=por" in command
    assert "title=Brazilian Portuguese" in command
    assert "title=English" in command
    assert "-disposition:1" in command
    assert command[command.index("-disposition:1") + 1] == "default"
    assert config.output.is_file()


def test_mux_source_audio_overrides_metadata(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    config = replace(config, language="de", track_name="  German Dub  ")
    commands: list[list[str]] = []
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)
    monkeypatch.setattr(
        "dubgraft.processing._validate_output",
        lambda path, *args, **kwargs: output_info(
            path, target_info, source_audio, language="ger", title="German Dub"
        ),
    )

    mux_source_audio(config, target_info, source_audio, offset=0)

    assert "language=ger" in commands[-1]
    assert "title=German Dub" in commands[-1]


def test_mux_source_audio_rejects_unknown_language(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    config = replace(config, language="invalid")
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    with pytest.raises(ProcessingError, match="invalid language code"):
        mux_source_audio(config, target_info, source_audio, offset=0)


@pytest.mark.parametrize(
    ("track_name", "error"),
    [("", "must not be empty"), ("bad\x00name", "null character")],
)
def test_mux_source_audio_rejects_invalid_track_name(
    track_name: str,
    error: str,
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    config = replace(config, track_name=track_name)
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    with pytest.raises(ProcessingError, match=error):
        mux_source_audio(config, target_info, source_audio, offset=0)


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("out_time_us=2500000", 2.5),
        ("out_time_ms=2500000", 2.5),
        ("out_time=00:00:02.500000", 2.5),
    ],
)
def test_progress_time_formats(timestamp: str, expected: float) -> None:
    lines = [f"{timestamp}\n", "progress=continue\n"]

    assert list(iter_progress_times(lines)) == [expected]


def test_ffmpeg_progress_is_monotonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    updates: list[tuple[str, float, float]] = []
    protocol = "".join(
        [
            "out_time_us=5000000\nprogress=continue\n",
            "out_time_us=4000000\nprogress=continue\n",
            "out_time_us=12000000\nprogress=end\n",
        ]
    )

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        commands.append(command)
        return _FakeProcess(protocol, kwargs["stderr"])  # type: ignore[arg-type]

    monkeypatch.setattr("dubgraft.execution.subprocess.Popen", fake_popen)

    _run_ffmpeg(
        ["ffmpeg", "-v", "error", "output.mkv"],
        "mux",
        expected_duration=10,
        progress=lambda *values: updates.append(values),
    )

    assert commands[0][1:4] == ["-nostats", "-progress", "pipe:1"]
    assert updates == [("mux", 5, 10), ("mux", 9.9, 10), ("mux", 10, 10)]


def test_ffmpeg_progress_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess(
            "progress=end\n",
            kwargs["stderr"],
            return_code=1,
            error="mux failed",
        ),
    )

    with pytest.raises(FFmpegError, match="mux failed"):
        _run_ffmpeg(
            ["ffmpeg", "output.mkv"],
            "mux",
            expected_duration=10,
            progress=lambda *args: None,
        )


def test_ffmpeg_progress_stops_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[_RunningProcess] = []

    def fake_popen(*args: object, **kwargs: object) -> _RunningProcess:
        process = _RunningProcess(
            "out_time_us=1000000\nprogress=continue\n",
            kwargs["stderr"],  # type: ignore[arg-type]
        )
        processes.append(process)
        return process

    monkeypatch.setattr("dubgraft.execution.subprocess.Popen", fake_popen)

    with pytest.raises(KeyboardInterrupt):
        _run_ffmpeg(
            ["ffmpeg", "output.mkv"],
            "mux",
            expected_duration=10,
            progress=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt),
        )

    assert processes[0].terminated
    assert processes[0].stdout.closed


def test_mux_source_audio_keeps_existing_output_when_ffmpeg_fails(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    config.output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "mux failed"),
    )

    with pytest.raises(FFmpegError, match="mux failed"):
        mux_source_audio(config, target_info, source_audio, offset=0)

    assert config.output.read_text(encoding="utf-8") == "existing"


def test_mux_source_audio_does_not_publish_over_a_late_output(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).touch()
        config.output.write_text("late output", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)
    monkeypatch.setattr(
        "dubgraft.processing._validate_output",
        lambda path, *args, **kwargs: output_info(path, target_info, source_audio),
    )

    with pytest.raises(ProcessingError, match="use --overwrite"):
        mux_source_audio(config, target_info, source_audio, offset=0)

    assert config.output.read_text(encoding="utf-8") == "late output"


@pytest.mark.parametrize(
    ("intercept", "filter_fragment"),
    [(0.25, "atrim=start=0.250000000"), (-0.25, "adelay=250.250250250:all=1")],
)
def test_mux_drift_audio_retimes_only_the_new_audio_stream(
    intercept: float,
    filter_fragment: str,
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    commands: list[list[str]] = []
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)
    monkeypatch.setattr(
        "dubgraft.processing._validate_reconstructed_audio", lambda *args: None
    )
    monkeypatch.setattr(
        "dubgraft.processing._validate_output",
        lambda path, *args, **kwargs: output_info(path, target_info, source_audio),
    )

    mux_drift_audio(
        config,
        target_info,
        source_audio,
        timeline(TimelineKind.DRIFT, slope=0.999, intercept=intercept),
    )

    assert len(commands) == 2
    reconstruction_command, command = commands
    filter_graph = reconstruction_command[
        reconstruction_command.index("-filter_complex") + 1
    ]
    assert filter_fragment in filter_graph
    assert "atempo=0.999000000000" in filter_graph
    assert "apad,atrim=duration=1000.000000000" in filter_graph
    assert reconstruction_command[reconstruction_command.index("-c:a") + 1] == "eac3"
    assert reconstruction_command[reconstruction_command.index("-b:a") + 1] == "640k"
    assert "reconstructed-audio.mka" in reconstruction_command[-1]
    assert command[command.index("-c") + 1] == "copy"
    assert "-filter_complex" not in command
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "1:0"
    assert command[command.index("-max_interleave_delta") + 1] == "0"
    assert config.output.is_file()


def test_output_validation_accepts_expected_mux(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    expected = output_info(config.output, target_info, source_audio)
    monkeypatch.setattr("dubgraft.processing.probe_media", lambda path: expected)

    validated = _validate_output(
        config.output,
        config,
        target_info,
        source_audio,
        audio_codec="eac3",
    )

    assert validated == replace(expected, added_audio_stream_index=4)


def test_output_validation_rejects_metadata_change(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    invalid = output_info(
        config.output, target_info, source_audio, language="eng"
    )
    monkeypatch.setattr("dubgraft.processing.probe_media", lambda path: invalid)

    with pytest.raises(
        ProcessingError, match="language expected 'por', found 'eng'.*not published"
    ):
        _validate_output(
            config.output,
            config,
            target_info,
            source_audio,
            audio_codec="eac3",
        )


def test_output_validation_rejects_chapter_and_disposition_changes(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    expected = output_info(config.output, target_info, source_audio)
    changed_streams = list(expected.streams)
    changed_streams[1] = replace(changed_streams[1], dispositions=())
    monkeypatch.setattr(
        "dubgraft.processing.probe_media",
        lambda path: replace(expected, streams=tuple(changed_streams)),
    )

    with pytest.raises(ProcessingError, match="dispositions expected"):
        _validate_output(
            config.output,
            config,
            target_info,
            source_audio,
            audio_codec="eac3",
        )

    monkeypatch.setattr(
        "dubgraft.processing.probe_media",
        lambda path: replace(expected, chapters=expected.chapters[:-1]),
    )
    with pytest.raises(ProcessingError, match="expected 2 chapters"):
        _validate_output(
            config.output,
            config,
            target_info,
            source_audio,
            audio_codec="eac3",
        )


def test_output_validation_rejects_added_audio_timing_change(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    expected = output_info(config.output, target_info, source_audio)
    streams = list(expected.streams)
    streams[-1] = replace(streams[-1], start_time=0.25, duration=900)
    monkeypatch.setattr(
        "dubgraft.processing.probe_media",
        lambda path: replace(expected, streams=tuple(streams)),
    )

    with pytest.raises(ProcessingError, match="added audio start expected"):
        _validate_output(
            config.output,
            config,
            target_info,
            source_audio,
            audio_codec="eac3",
            audio_start_time=0,
            audio_duration=999.9,
        )

    streams[-1] = replace(streams[-1], start_time=0, duration=900)
    with pytest.raises(ProcessingError, match="added audio duration expected"):
        _validate_output(
            config.output,
            config,
            target_info,
            source_audio,
            audio_codec="eac3",
            audio_start_time=0,
            audio_duration=999.9,
        )


def test_output_validation_identifies_reordered_added_audio(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    added_audio = replace(source_audio, index=10)
    reordered = MediaInfo(
        config.output,
        target_info.container,
        target_info.duration,
        None,
        None,
        (
            target_info.streams[0],
            added_audio,
            *target_info.streams[1:],
        ),
        target_info.title,
        target_info.chapters,
    )
    monkeypatch.setattr("dubgraft.processing.probe_media", lambda path: reordered)

    validated = _validate_output(
        config.output,
        config,
        target_info,
        source_audio,
        audio_codec="eac3",
    )

    assert validated.added_audio_stream_index == 10


def test_output_validation_rejects_hdr_side_data_change(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    target_streams = list(target_info.streams)
    target_streams[0] = replace(
        target_streams[0],
        video_side_data=(
            '{"dv_profile":5,"side_data_type":"DOVI configuration record"}',
        ),
    )
    target_info = replace(target_info, streams=tuple(target_streams))
    invalid = output_info(config.output, target_info, source_audio)
    invalid_streams = list(invalid.streams)
    invalid_streams[0] = replace(invalid_streams[0], video_side_data=())
    monkeypatch.setattr(
        "dubgraft.processing.probe_media",
        lambda path: replace(invalid, streams=tuple(invalid_streams)),
    )

    with pytest.raises(ProcessingError, match="video_side_data expected"):
        _validate_output(
            config.output,
            config,
            target_info,
            source_audio,
            audio_codec="eac3",
        )


def test_output_validation_accepts_mp4_cover_thumbnail_disposition(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    config = replace(config, output=config.output.with_suffix(".mp4"))
    cover = MediaStream(
        index=5,
        kind="video",
        codec="mjpeg",
        width=600,
        height=900,
        pixel_format="yuvj444p",
        attached_picture=True,
        dispositions=("attached_pic",),
    )
    target_info = replace(target_info, streams=(*target_info.streams, cover))
    expected = output_info(config.output, target_info, source_audio)
    streams = list(expected.streams)
    streams[-2] = replace(
        streams[-2],
        dispositions=("attached_pic", "timed_thumbnails"),
    )
    monkeypatch.setattr(
        "dubgraft.processing.probe_media",
        lambda path: replace(expected, streams=tuple(streams)),
    )

    validated = _validate_output(
        config.output,
        config,
        target_info,
        source_audio,
        audio_codec="eac3",
    )

    assert validated.added_audio_stream_index == len(target_info.streams)


def test_mux_preflight_suggests_target_extension(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, _, _ = processing_media
    config = replace(config, output=config.output.with_suffix(".mp4"))
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "dubgraft.processing._run_ffmpeg",
        lambda *args, **kwargs: (_ for _ in ()).throw(FFmpegError("unsupported codec")),
    )

    with pytest.raises(ProcessingError, match=r"Target extension \(.mkv\)"):
        validate_mux_compatibility(config, target_info)


def test_mux_preflight_preserves_mp4_dolby_vision(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, _, _ = processing_media
    config = replace(config, output=config.output.with_suffix(".mp4"))
    streams = list(target_info.streams)
    streams[0] = replace(streams[0], dolby_vision="Dolby Vision P5.0")
    target_info = replace(target_info, streams=tuple(streams))
    commands = []
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "dubgraft.processing._run_ffmpeg",
        lambda command, *args, **kwargs: commands.append(command),
    )

    validate_mux_compatibility(config, target_info)

    assert commands[0][commands[0].index("-strict") + 1] == "unofficial"


def test_mux_does_not_publish_failed_validation(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)
    monkeypatch.setattr(
        "dubgraft.processing._validate_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ProcessingError("generated Output failed validation")
        ),
    )

    with pytest.raises(ProcessingError, match="failed validation"):
        mux_source_audio(config, target_info, source_audio, offset=0)

    assert not config.output.exists()


def test_mux_drift_audio_rejects_unsupported_channel_count(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="at most 6.*avoid downmix"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, channels=8),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


@pytest.mark.parametrize(
    "profile", ["Dolby Digital Plus + Dolby Atmos", "E-AC-3 JOC"]
)
def test_mux_drift_audio_rejects_atmos_metadata_loss(
    profile: str,
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="would discard it"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, profile=profile),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


def test_mux_drift_audio_rejects_truehd_quality_loss(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="cannot preserve its lossless encoding"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, codec="truehd", profile=None, title="English"),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


def test_mux_drift_audio_rejects_atmos_title_metadata_loss(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="would discard it"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, profile=None, title="English Dolby Atmos"),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


@pytest.mark.parametrize("codec", ["flac", "alac", "pcm_s24le", "wavpack"])
def test_mux_drift_audio_rejects_lossless_conversion(
    codec: str,
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="would convert it to lossy audio"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, codec=codec),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


def test_mux_drift_audio_rejects_dts_hd_master_audio(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="would convert it to lossy audio"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, codec="dts", profile="DTS-HD MA"),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


@pytest.mark.parametrize("codec", ["ac4", "iamf", "mpegh_3d_audio"])
def test_mux_drift_audio_rejects_object_audio_codec(
    codec: str,
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="object-based or immersive"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, codec=codec),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


def test_mux_drift_audio_rejects_object_audio_side_data(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="object-based or immersive"):
        mux_drift_audio(
            config,
            target_info,
            replace(
                source_audio,
                audio_side_data=('{"side_data_type":"Dolby object audio metadata"}',),
            ),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


def test_mux_drift_audio_rejects_unclassified_codec(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="not classified as conventional lossy"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, codec="unknown"),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


def test_mux_drift_audio_rejects_unknown_channel_count(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="avoid an implicit downmix"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, channels=None),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


@pytest.mark.parametrize(
    ("changes", "detail"),
    [
        ({"channel_layout": None}, "channel layout"),
        ({"sample_rate": None}, "sample rate"),
    ],
)
def test_mux_drift_audio_rejects_unknown_audio_format(
    changes: dict[str, object],
    detail: str,
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match=detail):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, **changes),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


def test_reconstructed_audio_rejects_implicit_layout_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_audio = MediaStream(
        index=1,
        kind="audio",
        codec="aac",
        sample_rate=48_000,
        channels=6,
        channel_layout="6.0",
    )
    reconstructed_audio = MediaStream(
        index=0,
        kind="audio",
        codec="eac3",
        sample_rate=48_000,
        channels=5,
        channel_layout="5.0(side)",
    )
    monkeypatch.setattr(
        "dubgraft.processing.probe_media",
        lambda path: MediaInfo(path, "matroska,webm", 1, None, None, (reconstructed_audio,)),
    )

    with pytest.raises(ProcessingError, match="changed channel count.*conversion"):
        _validate_reconstructed_audio(tmp_path / "audio.mka", source_audio)


@pytest.mark.parametrize(
    ("kind", "median_offset", "expected_offset"),
    [
        (TimelineKind.DIRECT, 0.007, 0),
        (TimelineKind.STATIC, 2.5, 2.5),
    ],
)
def test_process_media_muxes_direct_and_static_timelines(
    kind: TimelineKind,
    median_offset: float,
    expected_offset: float,
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source_info, target_info, source_audio, target_audio = processing_media
    result = matching_result(kind, offset=median_offset)
    offsets = []
    matching_arguments = {}

    def match(*args: object, **kwargs: object) -> MatchingResult:
        matching_arguments.update(kwargs)
        return result

    monkeypatch.setattr("dubgraft.processing.match_audio_streams", match)
    monkeypatch.setattr(
        "dubgraft.processing.mux_source_audio",
        lambda *args, offset, **kwargs: offsets.append(offset),
    )

    assert (
        process_media(config, source_info, target_info, source_audio, target_audio)
        is result
    )
    assert offsets == [expected_offset]
    assert matching_arguments["source_duration"] == 999.9


def test_process_media_reports_inconclusive_timeline_without_muxing(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source_info, target_info, source_audio, target_audio = processing_media
    analysis = TimelineAnalysis(
        TimelineKind.INCONCLUSIVE,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        None,
        "no anchors were found",
    )
    result = MatchingResult((), (), 3, 20, analysis)
    monkeypatch.setattr(
        "dubgraft.processing.match_audio_streams", lambda *args, **kwargs: result
    )
    monkeypatch.setattr(
        "dubgraft.processing.mux_source_audio",
        lambda *args, **kwargs: pytest.fail("mux must not run"),
    )

    with pytest.raises(InconclusiveTimelineError, match="no anchors were found"):
        process_media(config, source_info, target_info, source_audio, target_audio)


def test_process_media_reconstructs_drift(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source_info, target_info, source_audio, target_audio = processing_media
    result = matching_result(TimelineKind.DRIFT, slope=0.999, intercept=0.003)
    analyses = []
    monkeypatch.setattr(
        "dubgraft.processing.match_audio_streams", lambda *args, **kwargs: result
    )
    monkeypatch.setattr(
        "dubgraft.processing.mux_drift_audio",
        lambda *args, **kwargs: analyses.append(args[-1]),
    )

    assert (
        process_media(config, source_info, target_info, source_audio, target_audio)
        is result
    )
    assert analyses == [result.timeline]


def test_process_media_requires_output_before_matching(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source_info, target_info, source_audio, target_audio = processing_media
    config = replace(config, output=None, analyze_only=True)
    monkeypatch.setattr(
        "dubgraft.processing.match_audio_streams",
        lambda *args, **kwargs: pytest.fail("matching must not run"),
    )

    with pytest.raises(ProcessingError, match="Output is required"):
        process_media(config, source_info, target_info, source_audio, target_audio)


@pytest.mark.parametrize("duration", [None, 0, float("nan"), float("inf")])
def test_process_media_requires_a_finite_target_duration(
    duration: float | None,
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
) -> None:
    config, source_info, target_info, source_audio, target_audio = processing_media

    with pytest.raises(ProcessingError, match="Target duration is unavailable"):
        process_media(
            config,
            source_info,
            replace(target_info, duration=duration),
            source_audio,
            target_audio,
        )
