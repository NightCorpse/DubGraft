import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from dubgraft.config import ProcessingConfig
from dubgraft.matching import MatchingResult, TimelineAnalysis, TimelineKind
from dubgraft.media import FFmpegError, MediaInfo, MediaStream
from dubgraft.processing import (
    InconclusiveTimelineError,
    ProcessingError,
    mux_drift_audio,
    mux_source_audio,
    process_media,
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
        channels=6,
        language="por",
        title="Brazilian Portuguese",
    )
    target_audio = MediaStream(index=1, kind="audio", codec="eac3")
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
    )
    config = ProcessingConfig(source, target, tmp_path / "output.mkv")
    return config, source_info, target_info, source_audio, target_audio


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
    commands: list[list[str]] = []
    monkeypatch.setattr("dubgraft.processing.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.processing.subprocess.run", fake_run)

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
    assert config.output.is_file()


def test_mux_source_audio_keeps_existing_output_when_ffmpeg_fails(
    processing_media: tuple[ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, target_info, source_audio, _ = processing_media
    config.output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr("dubgraft.processing.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "dubgraft.processing.subprocess.run",
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
    monkeypatch.setattr("dubgraft.processing.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).touch()
        config.output.write_text("late output", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.processing.subprocess.run", fake_run)

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
    monkeypatch.setattr("dubgraft.processing.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dubgraft.processing.subprocess.run", fake_run)

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


def test_mux_drift_audio_rejects_unsupported_channel_count(
    processing_media: tuple[
        ProcessingConfig, MediaInfo, MediaInfo, MediaStream, MediaStream
    ],
) -> None:
    config, _, target_info, source_audio, _ = processing_media

    with pytest.raises(ProcessingError, match="at most 6 audio channels"):
        mux_drift_audio(
            config,
            target_info,
            replace(source_audio, channels=8),
            timeline(TimelineKind.DRIFT, slope=0.999),
        )


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
        lambda *args, offset: offsets.append(offset),
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
        lambda *args: analyses.append(args[-1]),
    )

    assert (
        process_media(config, source_info, target_info, source_audio, target_audio)
        is result
    )
    assert analyses == [result.timeline]


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
