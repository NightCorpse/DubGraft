import json
from dataclasses import replace
from pathlib import Path

import pytest

from dubgraft import __version__
from dubgraft.cli import format_processing_summary, main, parse_processing_config
from dubgraft.config import ProcessingConfig
from dubgraft.matching import AudioMatch, MatchingResult, TimelineAnalysis, TimelineKind
from dubgraft.media import FFmpegError, MediaInfo, MediaProbeError, MediaStream
from dubgraft.processing import ProcessingError
from dubgraft.report import ReportError


@pytest.fixture
def available_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dubgraft.cli.validate_ffmpeg", lambda: None)
    monkeypatch.setattr("dubgraft.cli.validate_mux_compatibility", lambda *args: None)
    monkeypatch.setattr(
        "dubgraft.cli.validate_render_compatibility", lambda *args: None
    )

    def render(
        config: ProcessingConfig,
        target_info: MediaInfo,
        source_audio: MediaStream,
        *args: object,
        **kwargs: object,
    ) -> MediaInfo:
        assert config.output is not None
        added_audio = replace(
            source_audio,
            index=len(target_info.streams),
            language=config.language or source_audio.language,
            title=config.track_name or source_audio.title,
        )
        return MediaInfo(
            config.output,
            target_info.container,
            target_info.duration,
            None,
            None,
            (*target_info.streams, added_audio),
        )

    monkeypatch.setattr("dubgraft.cli.render_media", render)
    monkeypatch.setattr(
        "dubgraft.cli.analyze_media",
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


def test_cli_help_explains_inherited_output_extension(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "without an extension, uses the Target extension" in output


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


def test_cli_accepts_audio_metadata() -> None:
    config = parse_processing_config(
        [
            "source.mkv",
            "target.mkv",
            "output.mkv",
            "--language",
            "pt",
            "--track-name",
            "Português Brasileiro",
        ]
    )

    assert config.language == "por"
    assert config.track_name == "Português Brasileiro"


def test_cli_accepts_advanced_analysis_options() -> None:
    config = parse_processing_config(
        [
            "source.mkv",
            "target.mkv",
            "output.mkv",
            "--fingerprint-size",
            "8",
            "--scan-step",
            "10",
            "--search-radius",
            "90",
            "--min-confidence",
            "25",
            "--anchors",
            "8",
            "--anchor-gap",
            "120",
            "--jobs",
            "3",
            "--direct-limit",
            "35",
            "--force",
        ]
    )

    assert config.matching_config.fingerprint_size_seconds == 8
    assert config.matching_config.scan_step_seconds == 10
    assert config.matching_config.search_radius_seconds == 90
    assert config.matching_config.confidence_threshold == 25
    assert config.matching_config.anchor_count == 8
    assert config.matching_config.minimum_anchor_distance_seconds == 120
    assert config.matching_config.jobs == 3
    assert config.timeline_config.direct_tolerance_seconds == pytest.approx(0.035)
    assert config.force is True


def test_cli_accepts_short_advanced_analysis_options() -> None:
    config = parse_processing_config(
        [
            "source.mkv",
            "target.mkv",
            "output.mkv",
            "-F",
            "8",
            "-p",
            "10",
            "-r",
            "90",
            "-c",
            "25",
            "-a",
            "8",
            "-g",
            "120",
            "-j",
            "3",
            "-d",
            "35",
            "-f",
        ]
    )

    assert config.matching_config.fingerprint_size_seconds == 8
    assert config.matching_config.scan_step_seconds == 10
    assert config.matching_config.search_radius_seconds == 90
    assert config.matching_config.confidence_threshold == 25
    assert config.matching_config.anchor_count == 8
    assert config.matching_config.minimum_anchor_distance_seconds == 120
    assert config.matching_config.jobs == 3
    assert config.timeline_config.direct_tolerance_seconds == pytest.approx(0.035)
    assert config.force is True


def test_cli_rejects_fewer_than_four_anchors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_processing_config(
            ["source.mkv", "target.mkv", "output.mkv", "--anchors", "3"]
        )

    assert exit_info.value.code == 2
    assert "must be an integer of at least 4" in capsys.readouterr().err


@pytest.mark.parametrize("jobs", ["0", "-1", "invalid"])
def test_cli_rejects_invalid_matching_jobs(
    jobs: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_processing_config(
            ["source.mkv", "target.mkv", "output.mkv", "--jobs", jobs]
        )

    assert exit_info.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--min-confidence", "19", "use --force to continue"),
        ("--direct-limit", "51", "use --force to continue"),
    ],
)
def test_cli_explains_forced_analysis_limits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
    message: str,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(source),
                str(target),
                "--analyze-only",
                option,
                value,
            ]
        )

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value", "name"),
    [
        ("--fingerprint-size", "0", "fingerprint size"),
        ("--scan-step", "-1", "scan step"),
    ],
)
def test_cli_rejects_invalid_scan_parameters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
    name: str,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), "--analyze-only", option, value])

    assert exit_info.value.code == 2
    assert f"{name} must be a positive finite number" in capsys.readouterr().err


def test_cli_rejects_unknown_language(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_processing_config(
            [
                "source.mkv",
                "target.mkv",
                "output.mkv",
                "--language",
                "egn",
            ]
        )

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "did you mean 'eng'?" in error
    assert "src/dubgraft/data/iso-639-2.csv" in error


def test_cli_accepts_analyze_only_without_output() -> None:
    config = parse_processing_config(
        ["source.mkv", "target.mkv", "--analyze-only", "-S", "1", "-T", "4"]
    )

    assert config.output is None
    assert config.analyze_only is True
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


def test_cli_prints_default_progress(
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

    assert main([str(source), str(target), str(tmp_path / "output.mkv")]) == 0

    error = capsys.readouterr().err
    assert "Checking FFmpeg... done" in error
    assert "Inspecting Source... done" in error
    assert "Inspecting Target... done" in error
    assert "Analyzing alignment... done" in error
    assert "Rendering output... done" in error


def test_cli_quiet_suppresses_progress_and_summary(
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

    assert main([str(source), str(target), "--analyze-only", "--quiet"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_verbose_prints_processing_details(
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

    assert main([str(source), str(target), "--analyze-only", "--verbose"]) == 0

    error = capsys.readouterr().err
    assert f"Source: {source.resolve()}" in error
    assert "Source audio: [1] eac3" in error
    assert "Analysis: direct | 4/4 anchors" in error


def test_cli_formats_render_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()
    updates: list[str] = []
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)
    monkeypatch.setattr(
        "dubgraft.runtime.RunOutput.update",
        lambda self, detail: updates.append(detail),
    )

    def render(*args: object, **kwargs: object) -> None:
        progress = kwargs["progress"]
        progress("audio reconstruction", 5, 10)  # type: ignore[operator]
        progress("mux", 2.5, 10)  # type: ignore[operator]

    monkeypatch.setattr("dubgraft.cli.render_media", render)

    assert main([str(source), str(target), str(tmp_path / "output.mkv")]) == 0
    assert updates[-2:] == ["audio reconstruction (50%)", "mux (25%)"]


def test_cli_quiet_preserves_explicit_print_report(
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

    assert (
        main(
            [
                str(source),
                str(target),
                "--analyze-only",
                "--quiet",
                "--print-report",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("DubGraft Analysis Report")
    assert captured.err == ""


def test_cli_appends_detailed_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    log = tmp_path / "dubgraft.log"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    arguments = [
        str(source),
        str(target),
        "--analyze-only",
        "--quiet",
        "--log",
        str(log),
    ]
    assert main(arguments) == 0
    assert main(arguments) == 0

    contents = log.read_text(encoding="utf-8")
    assert contents.count("DubGraft run started") == 2
    assert "Analyzing alignment started" in contents
    assert f"Source: {source.resolve()}" in contents


def test_log_captures_unexpected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    log = tmp_path / "failure.log"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    def fail_analysis(*args: object, **kwargs: object) -> MatchingResult:
        raise RuntimeError("unexpected test failure")

    monkeypatch.setattr("dubgraft.cli.analyze_media", fail_analysis)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(source),
                str(target),
                "--analyze-only",
                "--log",
                str(log),
            ]
        )

    assert exit_info.value.code == 1
    error = capsys.readouterr().err
    assert "unexpected failure: RuntimeError: unexpected test failure" in error
    assert f"details written to {log.resolve()}" in error
    contents = log.read_text(encoding="utf-8")
    assert "ERROR dubgraft: Unexpected failure" in contents
    assert "Traceback (most recent call last)" in contents
    assert "RuntimeError: unexpected test failure" in contents


def test_log_captures_configuration_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "missing.mkv"
    target = tmp_path / "target.mkv"
    log = tmp_path / "configuration.log"
    target.touch()

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(source),
                str(target),
                str(tmp_path / "output.mkv"),
                "--log",
                str(log),
            ]
        )

    assert exit_info.value.code == 2
    assert "Source file does not exist" in capsys.readouterr().err
    contents = log.read_text(encoding="utf-8")
    assert "ERROR dubgraft: Source file does not exist" in contents


def test_interrupt_returns_130(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    log = tmp_path / "interrupted.log"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    def interrupt(*args: object, **kwargs: object) -> MatchingResult:
        raise KeyboardInterrupt

    monkeypatch.setattr("dubgraft.cli.analyze_media", interrupt)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(source),
                str(target),
                "--analyze-only",
                "--log",
                str(log),
            ]
        )

    assert exit_info.value.code == 130
    assert "interrupted by user" in capsys.readouterr().err
    contents = log.read_text(encoding="utf-8")
    assert "Analyzing alignment failed" in contents
    assert "WARNING dubgraft: Run cancelled by user" in contents


def test_cli_rejects_verbose_and_quiet_together(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_processing_config(
            ["source.mkv", "target.mkv", "--analyze-only", "--verbose", "--quiet"]
        )

    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


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


def test_cli_rejects_incompatible_output_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.validate_ffmpeg", lambda: None)
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)
    monkeypatch.setattr(
        "dubgraft.cli.validate_mux_compatibility",
        lambda *args: (_ for _ in ()).throw(
            ProcessingError("Output container cannot preserve every Target stream")
        ),
    )
    analyzed = False

    def analyze(*args: object, **kwargs: object) -> MatchingResult:
        nonlocal analyzed
        analyzed = True
        return processing_result(TimelineKind.DIRECT)

    monkeypatch.setattr("dubgraft.cli.analyze_media", analyze)

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(tmp_path / "output.mp4")])

    assert exit_info.value.code == 1
    assert analyzed is False
    assert "cannot preserve every Target stream" in capsys.readouterr().err


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

    monkeypatch.setattr("dubgraft.cli.analyze_media", lambda *args, **kwargs: result)

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(tmp_path / "output.mkv")])

    assert exit_info.value.code == 1
    error = capsys.readouterr().err
    assert "inconclusive analysis: no anchors were found" in error
    assert "Anchors: 0/4 | coverage: 0.0%" in error
    assert "No output was created." in error


def test_analyze_only_does_not_process_media(
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
    monkeypatch.setattr(
        "dubgraft.cli.analyze_media",
        lambda *args, **kwargs: processing_result(TimelineKind.DIRECT),
    )
    monkeypatch.setattr(
        "dubgraft.cli.render_media",
        lambda *args, **kwargs: pytest.fail("processing must not run"),
    )

    assert main([str(source), str(target), "--analyze-only"]) == 0

    output = capsys.readouterr().out
    assert "Analysis: direct" in output
    assert "No output was created (--analyze-only)." in output


def test_cli_writes_detailed_json_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    report = tmp_path / "analysis-without-extension"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    assert (
        main(
            [
                str(source),
                str(target),
                "--analyze-only",
                "--report",
                str(report),
            ]
        )
        == 0
    )

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["status"] == "analysis_only"
    assert data["mode"] == "analyze_only"
    assert data["media"]["output"] is None
    assert data["matching"]["anchor_count"] == 4
    assert len(data["matching"]["anchors"]) == 4
    assert data["timeline"]["classification"] == "direct"
    assert data["formulas"]["timeline"].startswith("source_time =")
    assert data["processing"]["rendered"] is False
    assert data["processing"]["output_validated"] is False
    assert data["processing"]["target_streams_preserved"] is None
    assert data["processing"]["added_audio"] is None


def test_cli_prints_full_report_after_summary(
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

    assert (
        main(
            [
                str(source),
                str(target),
                str(tmp_path / "output.mkv"),
                "--print-report",
                "--language",
                "pt",
                "--track-name",
                "Dublado",
                "--fingerprint-size",
                "8",
                "--scan-step",
                "10",
                "--search-radius",
                "90",
                "--min-confidence",
                "25",
                "--anchors",
                "4",
                "--anchor-gap",
                "10",
                "--direct-limit",
                "35",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.index("Analysis: direct") < output.index("DubGraft Analysis Report")
    assert "Formulas" in output
    assert "Timeline: source_time = slope * target_time + intercept" in output
    assert "Candidates (4)" in output
    assert "Selected anchors (4)" in output
    assert "Target (s)" in output
    assert "Output validated: yes" in output
    assert "Requested metadata overrides: language=por | title=Dublado" in output
    assert "Added audio: stream 1 | eac3 | por | Dublado" in output
    assert "Fingerprint size: 8.000s" in output
    assert "Scan step: 10.000s" in output
    assert "Search radius: 90.000s" in output
    assert "Minimum confidence: 25.000" in output
    assert "Matching jobs: automatic (up to 4)" in output
    assert "Direct limit: 35.000ms" in output


def test_cli_writes_completed_processing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    output = tmp_path / "output.mkv"
    report = tmp_path / "report.json"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    assert (
        main(
            [
                str(source),
                str(target),
                str(output),
                "--report",
                str(report),
                "--language",
                "pt",
                "--track-name",
                "Português Brasileiro",
                "--fingerprint-size",
                "8",
                "--scan-step",
                "10",
                "--search-radius",
                "90",
                "--min-confidence",
                "25",
                "--anchors",
                "4",
                "--anchor-gap",
                "10",
                "--direct-limit",
                "35",
            ]
        )
        == 0
    )

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["mode"] == "process"
    assert data["media"]["output"] == str(output.resolve())
    assert data["processing"]["rendered"] is True
    assert data["processing"]["output_validated"] is True
    assert data["processing"]["target_streams_preserved"] == 1
    assert data["processing"]["requested_metadata_overrides"] == {
        "language": "por",
        "title": "Português Brasileiro",
    }
    assert data["processing"]["added_audio"]["language"] == "por"
    assert data["processing"]["added_audio"]["title"] == "Português Brasileiro"
    assert data["parameters"]["matching"]["fingerprint_size_seconds"] == 8
    assert data["parameters"]["matching"]["scan_step_seconds"] == 10
    assert data["parameters"]["matching"]["search_radius_seconds"] == 90
    assert data["parameters"]["matching"]["confidence_threshold"] == 25
    assert data["parameters"]["matching"]["anchor_count"] == 4
    assert data["parameters"]["matching"]["minimum_anchor_distance_seconds"] == 10
    assert data["parameters"]["matching"]["jobs"] is None
    assert data["parameters"]["timeline"]["direct_tolerance_seconds"] == 0.035


def test_cli_preserves_analysis_report_when_rendering_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    report = tmp_path / "failed.json"
    source.touch()
    target.touch()
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    def fail_render(*args: object, **kwargs: object) -> None:
        raise ProcessingError("Atmos cannot be preserved")

    monkeypatch.setattr("dubgraft.cli.render_media", fail_render)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(source),
                str(target),
                str(tmp_path / "output.mkv"),
                "--report",
                str(report),
                "--print-report",
            ]
        )

    assert exit_info.value.code == 1
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "processing_failed"
    assert data["error"] == "Atmos cannot be preserved"
    assert data["processing"]["output_validated"] is False
    assert data["processing"]["target_streams_preserved"] is None
    assert data["processing"]["added_audio"] is None
    captured = capsys.readouterr()
    assert "Status: processing_failed" in captured.out
    assert "Target streams preserved: not validated" in captured.out
    assert "Atmos cannot be preserved" in captured.err


def test_cli_reports_partial_success_when_json_write_fails(
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
    monkeypatch.setattr("dubgraft.cli.probe_media", single_audio_info)

    def fail_report(*args: object, **kwargs: object) -> None:
        raise ReportError("could not write report")

    monkeypatch.setattr("dubgraft.cli.write_json_report", fail_report)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(source),
                str(target),
                str(output),
                "--report",
                str(tmp_path / "report.json"),
                "--print-report",
            ]
        )

    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert "DubGraft Analysis Report" in captured.out
    assert f"Output was created at {output.resolve()}" in captured.err


def test_analyze_only_writes_and_prints_inconclusive_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available_ffmpeg: None,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    report = tmp_path / "inconclusive.json"
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
    monkeypatch.setattr(
        "dubgraft.cli.analyze_media",
        lambda *args, **kwargs: MatchingResult((), (), 4, 20, analysis),
    )

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(source),
                str(target),
                "--analyze-only",
                "--report",
                str(report),
                "--print-report",
            ]
        )

    assert exit_info.value.code == 1
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "inconclusive"
    captured = capsys.readouterr()
    assert "Status: inconclusive" in captured.out
    assert "No output was created." in captured.err


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
    output_info = MediaInfo(
        config.output,
        target_info.container,
        target_info.duration,
        None,
        None,
        (
            *target_info.streams,
            replace(source_audio, index=1, language="por", title="Dublado"),
        ),
    )

    summary = format_processing_summary(
        result, config, target_info, source_audio, output_info
    )

    assert "Anchors: 4/4 | coverage: 80.0%" in summary
    assert "Target streams: 1 preserved" in summary
    assert "Added audio: [1] eac3" in summary
    assert "| por | Dublado" in summary
    assert "Output validation: passed" in summary
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
    assert "Available Source audio streams:" in error
    assert "[1] aac | por" in error
    assert "[2] eac3 | eng" in error
    assert "Use -S INDEX or --source-audio INDEX" in error
    assert "could not select Target audio" in error
    assert "Available Target audio streams:" in error
    assert "Use -T INDEX or --target-audio INDEX" in error


def test_cli_suggests_target_option_when_only_target_is_ambiguous(
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
        stream_count = 1 if path == source.resolve() else 2
        return MediaInfo(
            path=path,
            container="matroska,webm",
            duration=None,
            size=None,
            bit_rate=None,
            streams=tuple(
                MediaStream(index=index, kind="audio", codec="eac3")
                for index in range(1, stream_count + 1)
            ),
        )

    monkeypatch.setattr("dubgraft.cli.probe_media", probe)

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(tmp_path / "output.mkv")])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "could not select Source audio" not in error
    assert "could not select Target audio" in error
    assert "Use -T INDEX or --target-audio INDEX" in error


def test_cli_suggests_missing_source_metadata_options(
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

    assert main([str(source), str(target), str(tmp_path / "output.mkv")]) == 0

    error = capsys.readouterr().err
    assert "use --language CODE to set it" in error
    assert "use --track-name NAME to set it" in error


def test_cli_does_not_suggest_overridden_source_metadata(
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

    assert (
        main(
            [
                str(source),
                str(target),
                str(tmp_path / "output.mkv"),
                "--language",
                "pt",
                "--track-name",
                "Dublado",
            ]
        )
        == 0
    )

    assert "warning:" not in capsys.readouterr().err


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
    error = capsys.readouterr().err
    assert error.count("ffmpeg was not found in PATH") == 1


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
    assert (
        f"Output must not be the {input_name.title()} file" in capsys.readouterr().err
    )


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
