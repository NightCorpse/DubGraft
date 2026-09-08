from pathlib import Path

import pytest

from dubgraft import __version__
from dubgraft.cli import format_processing_summary, main, parse_processing_config
from dubgraft.config import ProcessingConfig
from dubgraft.matching import AudioMatch, MatchingResult, TimelineAnalysis, TimelineKind
from dubgraft.media import FFmpegError, MediaInfo, MediaProbeError, MediaStream
from dubgraft.processing import InconclusiveTimelineError


@pytest.fixture
def available_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dubgraft.cli.validate_ffmpeg", lambda: None)
    monkeypatch.setattr(
        "dubgraft.cli.process_media",
        lambda *args, **kwargs: processing_result(TimelineKind.DIRECT),
    )


def processing_result(
    kind: TimelineKind,
    *,
    offset: float = 0.007,
    slope: float = 1.0,
    fallback: bool = False,
) -> MatchingResult:
    anchors = tuple(AudioMatch(time, time, 0, 100) for time in (100, 400, 700, 900))
    timeline = TimelineAnalysis(
        kind,
        offset,
        slope,
        offset,
        0.001,
        0.002,
        0.8,
        1.0 if kind is not TimelineKind.DRIFT else 0.0,
        (slope - 1) * 1000,
        None,
        fallback,
        0.004 if fallback else None,
    )
    return MatchingResult(anchors, anchors, 4, 20, timeline)


def single_audio_info(path: Path) -> MediaInfo:
    return MediaInfo(
        path=path,
        container="matroska,webm",
        duration=None,
        size=None,
        bit_rate=None,
        streams=(MediaStream(index=1, kind="audio", codec="eac3"),),
    )


def test_cli_without_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "usage: dubgraft" in output
    assert (
        "Align and graft dubbed audio across media releases using distributed audio anchors."
        in " ".join(output.split())
    )


def test_cli_reports_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"dubgraft {__version__}"


@pytest.mark.parametrize(
    "arguments",
    [
        ["source.mkv", "target.mkv", "output.mkv"],
        ["source.mkv", "target.mkv", "-o", "output.mkv"],
        ["-s", "source.mkv", "-t", "target.mkv", "-o", "output.mkv"],
    ],
)
def test_cli_accepts_all_processing_forms(arguments: list[str]) -> None:
    assert parse_processing_config(arguments) == ProcessingConfig(
        source=Path("source.mkv"),
        target=Path("target.mkv"),
        output=Path("output.mkv"),
    )


def test_cli_accepts_long_audio_stream_options() -> None:
    config = parse_processing_config(
        [
            "source.mkv",
            "target.mkv",
            "output.mkv",
            "--source-audio",
            "1",
            "--target-audio",
            "4",
        ]
    )

    assert config.source_audio_index == 1
    assert config.target_audio_index == 4


def test_cli_accepts_short_audio_options_between_media_paths() -> None:
    config = parse_processing_config(
        ["source.mkv", "-S", "1", "target.mkv", "-T", "4", "output.mkv"]
    )

    assert config.source == Path("source.mkv")
    assert config.source_audio_index == 1
    assert config.target == Path("target.mkv")
    assert config.target_audio_index == 4
    assert config.output == Path("output.mkv")


@pytest.mark.parametrize("value", ["-1", "audio"])
def test_cli_rejects_invalid_audio_stream_index(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_processing_config(
            [
                "source.mkv",
                "target.mkv",
                "output.mkv",
                "--source-audio",
                value,
            ]
        )

    assert exit_info.value.code == 2
    assert "must be a non-negative integer" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "role"),
    [("--source", "SOURCE"), ("--target", "TARGET"), ("--output", "OUTPUT")],
)
def test_cli_rejects_duplicate_roles(
    option: str, role: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_processing_config(
            ["source.mkv", "target.mkv", "output.mkv", option, "duplicate.mkv"]
        )

    assert exit_info.value.code == 2
    assert f"{role} was provided more than once" in capsys.readouterr().err


def test_cli_requires_all_processing_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_processing_config(["source.mkv", "target.mkv"])

    assert exit_info.value.code == 2
    assert "OUTPUT is required" in capsys.readouterr().err


def test_cli_accepts_valid_media_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available_ffmpeg: None
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    output = tmp_path / "output.mkv"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    assert main([str(source), str(target), str(output)]) == 0


def test_cli_rejects_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "missing.mkv"
    target = tmp_path / "target.mkv"
    target.touch()

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(tmp_path / "output.mkv")])

    assert exit_info.value.code == 2
    assert "Source file does not exist" in capsys.readouterr().err


def test_cli_requires_overwrite_for_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    output = tmp_path / "output.mkv"
    source.touch()
    target.touch()
    output.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(output)])

    assert exit_info.value.code == 2
    assert "use --overwrite to replace it" in capsys.readouterr().err
    assert main([str(source), str(target), str(output), "--overwrite"]) == 0


def test_cli_selects_explicit_audio_stream_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available_ffmpeg: None
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    def probe(path: Path) -> MediaInfo:
        return MediaInfo(
            path=path,
            container="matroska,webm",
            duration=None,
            size=None,
            bit_rate=None,
            streams=(
                MediaStream(index=1, kind="audio", codec="aac"),
                MediaStream(index=3, kind="audio", codec="eac3"),
            ),
        )

    monkeypatch.setattr("dubgraft.cli.probe_media", probe)

    assert (
        main(
            [
                str(source),
                "-S",
                "1",
                str(target),
                "-T",
                "3",
                str(tmp_path / "output.mkv"),
            ]
        )
        == 0
    )


def test_cli_reports_inconclusive_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)
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
    result = MatchingResult((), (), 4, 20, analysis)

    def fail_processing(*args: object, **kwargs: object) -> None:
        raise InconclusiveTimelineError(analysis, result)

    monkeypatch.setattr("dubgraft.cli.process_media", fail_processing)

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(tmp_path / "output.mkv")])

    assert exit_info.value.code == 1
    error = capsys.readouterr().err
    assert "inconclusive analysis: no anchors were found" in error
    assert "Anchors: 0/4 | coverage: 0.0%" in error
    assert "No output was created." in error


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            processing_result(TimelineKind.DIRECT),
            ("Analysis: direct", "Offset: +0.007s", "copied without re-encoding"),
        ),
        (
            processing_result(TimelineKind.STATIC, offset=1.25),
            (
                "Analysis: static",
                "trimmed Source beginning by 1.250s",
                "copied without re-encoding",
            ),
        ),
        (
            processing_result(
                TimelineKind.DRIFT,
                offset=-0.5,
                slope=0.999,
                fallback=True,
            ),
            (
                "Analysis: drift (strict duration fallback)",
                "total drift -1.000s",
                "Fallback duration error: 0.004s",
                "E-AC-3 640 kb/s, 5.1(side), re-encoded once",
            ),
        ),
    ],
)
def test_processing_summary_reports_decisions(
    result: MatchingResult,
    expected: tuple[str, ...],
    tmp_path: Path,
) -> None:
    source_audio = MediaStream(
        index=1,
        kind="audio",
        codec="eac3",
        channels=6,
        channel_layout="5.1(side)",
    )
    target_info = MediaInfo(
        tmp_path / "target.mkv",
        "matroska,webm",
        1000,
        None,
        None,
        (MediaStream(index=0, kind="video", codec="hevc"),),
    )
    config = ProcessingConfig(
        tmp_path / "source.mkv",
        target_info.path,
        tmp_path / "output.mkv",
    )

    summary = format_processing_summary(result, config, target_info, source_audio)

    assert "Anchors: 4/4 | coverage: 80.0%" in summary
    assert "Target streams: 1 preserved" in summary
    assert f"Created: {config.output}" in summary
    for line in expected:
        assert line in summary


def test_cli_lists_audio_candidates_when_selection_is_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    def probe(path: Path) -> MediaInfo:
        return MediaInfo(
            path=path,
            container="matroska,webm",
            duration=None,
            size=None,
            bit_rate=None,
            streams=(
                MediaStream(index=1, kind="audio", codec="aac", language="por"),
                MediaStream(index=2, kind="audio", codec="eac3", language="eng"),
            ),
        )

    monkeypatch.setattr("dubgraft.cli.probe_media", probe)

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(tmp_path / "output.mkv")])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "could not select Source audio" in error
    assert "[1] aac | por" in error
    assert "[2] eac3 | eng" in error
    assert "Use -S INDEX or --source-audio INDEX" in error


def test_cli_reports_unavailable_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    def fail_validation() -> None:
        raise FFmpegError("ffmpeg was not found in PATH")

    monkeypatch.setattr("dubgraft.cli.validate_ffmpeg", fail_validation)

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(tmp_path / "output.mkv")])

    assert exit_info.value.code == 1
    assert "ffmpeg was not found in PATH" in capsys.readouterr().err


@pytest.mark.parametrize("input_name", ["source", "target"])
def test_cli_never_allows_output_to_replace_an_input(
    input_name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()
    output = source if input_name == "source" else target

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(output), "--overwrite"])

    assert exit_info.value.code == 2
    assert f"Output must not be the {input_name.title()} file" in capsys.readouterr().err


def test_cli_rejects_missing_output_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()
    output = tmp_path / "missing" / "output.mkv"

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(output)])

    assert exit_info.value.code == 2
    assert "Output directory does not exist" in capsys.readouterr().err


def test_cli_inspect_prints_video_audio_and_other_stream_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    media = tmp_path / "episode.mkv"
    info = MediaInfo(
        path=media,
        container="matroska,webm",
        duration=3217.952,
        size=1_994_237_191,
        bit_rate=4_957_779,
        streams=(
            MediaStream(
                index=0,
                kind="video",
                codec="hevc",
                profile="Main 10",
                width=3840,
                height=1920,
                frame_rate=24000 / 1001,
                pixel_format="yuv420p10le",
                color_transfer="smpte2084",
                default=True,
                dolby_vision="Dolby Vision P8.1",
            ),
            MediaStream(
                index=1,
                kind="audio",
                codec="eac3",
                profile="Dolby Digital Plus + Dolby Atmos",
                sample_rate=48000,
                channels=6,
                channel_layout="5.1(side)",
                bit_rate=640000,
                language="eng",
                default=True,
            ),
            MediaStream(index=2, kind="subtitle", codec="subrip"),
            MediaStream(index=3, kind="subtitle", codec="subrip"),
        ),
    )
    monkeypatch.setattr("dubgraft.cli.probe_media", lambda path: info)

    assert main(["inspect", str(media)]) == 0

    output = capsys.readouterr().out
    assert "Media: episode.mkv" in output
    assert "[0] hevc Main 10 | 3840x1920 | 23.976 fps" in output
    assert "Dolby Vision P8.1 / HDR10" in output
    assert "[1] eac3 Atmos | 5.1(side) | 48 kHz | 640 kb/s | eng | default" in output
    assert "2 subtitles (preserved)" in output
    assert "subrip" not in output


def test_cli_inspect_reports_probe_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_probe(path: Path) -> MediaInfo:
        raise MediaProbeError("ffprobe was not found in PATH")

    monkeypatch.setattr("dubgraft.cli.probe_media", fail_probe)

    assert main(["inspect", "episode.mkv"]) == 1

    assert "ffprobe was not found in PATH" in capsys.readouterr().err


def test_cli_inspect_prints_multiple_media_in_argument_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = [Path("source.mkv"), Path("target.mkv"), Path("extra.mkv")]

    def probe(path: Path) -> MediaInfo:
        return MediaInfo(path, "matroska,webm", None, None, None, ())

    monkeypatch.setattr("dubgraft.cli.probe_media", probe)

    assert main(["inspect", *(str(path) for path in paths)]) == 0

    output = capsys.readouterr().out
    positions = [output.index(f"Media: {path.name}") for path in paths]
    assert positions == sorted(positions)
    assert output.count("\n\nMedia:") == 2


def test_cli_inspect_continues_after_individual_probe_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inspected = []

    def probe(path: Path) -> MediaInfo:
        inspected.append(path)
        if path == Path("broken.mkv"):
            raise MediaProbeError("invalid media data")
        return MediaInfo(path, "matroska,webm", None, None, None, ())

    monkeypatch.setattr("dubgraft.cli.probe_media", probe)

    assert main(["inspect", "source.mkv", "broken.mkv", "target.mkv"]) == 1

    captured = capsys.readouterr()
    assert "Media: source.mkv" in captured.out
    assert "Media: target.mkv" in captured.out
    assert "broken.mkv: invalid media data" in captured.err
    assert inspected == [Path("source.mkv"), Path("broken.mkv"), Path("target.mkv")]


def test_cli_inspect_requires_at_least_one_media(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["inspect"])

    assert exit_info.value.code == 2
    assert "the following arguments are required: MEDIA" in capsys.readouterr().err


def test_cli_inspect_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", "--help"])

    assert exit_info.value.code == 0
    assert "usage: dubgraft inspect" in capsys.readouterr().out
