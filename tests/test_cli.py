from pathlib import Path

import pytest

from dubgraft import __version__
from dubgraft.cli import main, parse_processing_config
from dubgraft.config import ProcessingConfig
from dubgraft.media import MediaInfo, MediaProbeError, MediaStream


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


def test_cli_accepts_valid_media_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    output = tmp_path / "output.mkv"
    source.touch()
    target.touch()

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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    output = tmp_path / "output.mkv"
    source.touch()
    target.touch()
    output.touch()

    with pytest.raises(SystemExit) as exit_info:
        main([str(source), str(target), str(output)])

    assert exit_info.value.code == 2
    assert "use --overwrite to replace it" in capsys.readouterr().err
    assert main([str(source), str(target), str(output), "--overwrite"]) == 0


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

    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", "episode.mkv"])

    assert exit_info.value.code == 1
    assert "ffprobe was not found in PATH" in capsys.readouterr().err


def test_cli_inspect_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", "--help"])

    assert exit_info.value.code == 0
    assert "usage: dubgraft inspect" in capsys.readouterr().out
