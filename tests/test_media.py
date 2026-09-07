import json
import subprocess
from pathlib import Path

import pytest

from dubgraft.media import MediaProbeError, probe_media


def test_probe_media_reads_ffprobe_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mkv"
    media.touch()
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "hevc",
                "profile": "Main 10",
                "width": 3840,
                "height": 1920,
                "pix_fmt": "yuv420p10le",
                "color_transfer": "smpte2084",
                "avg_frame_rate": "24000/1001",
                "disposition": {"default": 1},
                "side_data_list": [
                    {
                        "side_data_type": "DOVI configuration record",
                        "dv_profile": 8,
                        "dv_bl_signal_compatibility_id": 1,
                    }
                ],
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "eac3",
                "profile": "Dolby Digital Plus + Dolby Atmos",
                "sample_rate": "48000",
                "channels": 6,
                "channel_layout": "5.1(side)",
                "bit_rate": "640000",
                "tags": {"language": "eng", "title": "English"},
            },
            {"index": 2, "codec_type": "subtitle", "codec_name": "subrip"},
        ],
        "format": {
            "format_name": "matroska,webm",
            "duration": "3217.952",
            "size": "1994237191",
            "bit_rate": "4957779",
        },
    }
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr("dubgraft.media.shutil.which", lambda name: "/bin/ffprobe")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr("dubgraft.media.subprocess.run", fake_run)

    info = probe_media(media)

    assert info.path == media.resolve()
    assert info.container == "matroska,webm"
    assert info.duration == pytest.approx(3217.952)
    assert info.size == 1994237191
    assert len(info.streams) == 3
    assert info.streams[0].frame_rate == pytest.approx(24000 / 1001)
    assert info.streams[0].dolby_vision == "Dolby Vision P8.1"
    assert info.streams[1].language == "eng"
    assert info.streams[1].channels == 6
    command, options = calls[0]
    assert command[0] == "/bin/ffprobe"
    assert command[-1] == str(media.resolve())
    assert "-of" in command
    assert "shell" not in options
    assert options["check"] is False


def test_probe_media_requires_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mkv"
    media.touch()
    monkeypatch.setattr("dubgraft.media.shutil.which", lambda name: None)

    with pytest.raises(MediaProbeError, match="ffprobe was not found in PATH"):
        probe_media(media)


def test_probe_media_reports_ffprobe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "broken.mkv"
    media.touch()
    monkeypatch.setattr("dubgraft.media.shutil.which", lambda name: "/bin/ffprobe")
    monkeypatch.setattr(
        "dubgraft.media.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "Invalid data"),
    )

    with pytest.raises(MediaProbeError, match="Invalid data"):
        probe_media(media)


def test_probe_media_rejects_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "broken.mkv"
    media.touch()
    monkeypatch.setattr("dubgraft.media.shutil.which", lambda name: "/bin/ffprobe")
    monkeypatch.setattr(
        "dubgraft.media.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "not json", ""),
    )

    with pytest.raises(MediaProbeError, match="ffprobe returned invalid JSON"):
        probe_media(media)
