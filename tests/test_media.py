import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dubgraft.media import (
    AudioSelectionError,
    FFmpegError,
    FFmpegInfo,
    MediaChapter,
    MediaInfo,
    MediaProbeError,
    MediaStream,
    _stream_title,
    extract_audio_window,
    probe_media,
    select_audio_stream,
    validate_ffmpeg,
)


def media_info(*streams: MediaStream) -> MediaInfo:
    return MediaInfo(
        path=Path("episode.mkv"),
        container="matroska,webm",
        duration=None,
        size=None,
        bit_rate=None,
        streams=streams,
    )


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"title": "English"}, "English"),
        ({"name": "English"}, "English"),
        ({"handler_name": "English"}, "English"),
        ({"handler_name": "SoundHandler"}, None),
        ({"handler_name": "VideoHandler"}, None),
    ],
)
def test_stream_title_reads_explicit_metadata(
    tags: dict[str, str], expected: str | None
) -> None:
    assert _stream_title(tags) == expected


def test_validate_ffmpeg_reads_version(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ffmpeg version 9.0\n", "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)

    assert validate_ffmpeg() == FFmpegInfo(Path("/bin/ffmpeg"), "ffmpeg version 9.0")
    command, options = calls[0]
    assert command == ["/bin/ffmpeg", "-version"]
    assert options["timeout"] == 10
    assert options["check"] is False
    assert "shell" not in options


def test_validate_ffmpeg_requires_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: None)

    with pytest.raises(FFmpegError, match="ffmpeg was not found in PATH"):
        validate_ffmpeg()


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (subprocess.CompletedProcess([], 1, "", "broken install"), "broken install"),
        (subprocess.CompletedProcess([], 0, "", ""), "no version information"),
    ],
)
def test_validate_ffmpeg_rejects_invalid_response(
    result: subprocess.CompletedProcess[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run", lambda *args, **kwargs: result
    )

    with pytest.raises(FFmpegError, match=message):
        validate_ffmpeg()


def test_extract_audio_window_reads_selected_stream_as_normalized_mono_pcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mkv"
    media.touch()
    pcm = np.array([-32768, 0, 16384, 32767], dtype=np.int16).tobytes()
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, pcm, b"")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)

    samples = extract_audio_window(media, 3, 12.5, 6)

    assert samples.dtype == np.float32
    assert samples == pytest.approx([-1.0, 0.0, 0.5, 32767 / 32768])
    command, options = calls[0]
    assert command[command.index("-map") + 1] == "0:3"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "22050"
    assert command[-1] == "pipe:1"
    assert options["timeout"] == 60
    assert "shell" not in options


def test_extract_audio_window_reports_ffmpeg_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "broken.mkv"
    media.touch()
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, b"", b"invalid audio stream"
        ),
    )

    with pytest.raises(FFmpegError, match="invalid audio stream"):
        extract_audio_window(media, 4, 0, 6)


def test_extract_audio_window_returns_empty_samples_after_media_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mkv"
    media.touch()
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, b"", b""),
    )

    samples = extract_audio_window(media, 1, 4000, 6)

    assert samples.dtype == np.float32
    assert samples.size == 0


def test_select_audio_stream_selects_the_only_audio_automatically() -> None:
    audio = MediaStream(index=2, kind="audio", codec="eac3")

    assert select_audio_stream(media_info(audio)) is audio


def test_select_audio_stream_uses_global_stream_index() -> None:
    video = MediaStream(index=0, kind="video", codec="hevc")
    first_audio = MediaStream(index=1, kind="audio", codec="aac")
    second_audio = MediaStream(index=4, kind="audio", codec="eac3")

    assert (
        select_audio_stream(media_info(video, first_audio, second_audio), index=4)
        is second_audio
    )


def test_select_audio_stream_reports_ambiguous_audio_candidates() -> None:
    audios = (
        MediaStream(index=1, kind="audio", codec="aac"),
        MediaStream(index=2, kind="audio", codec="eac3"),
    )

    with pytest.raises(AudioSelectionError, match="multiple audio streams") as error:
        select_audio_stream(media_info(*audios))

    assert error.value.candidates == audios


def test_select_audio_stream_rejects_non_audio_index() -> None:
    video = MediaStream(index=0, kind="video", codec="hevc")
    audio = MediaStream(index=1, kind="audio", codec="eac3")

    with pytest.raises(AudioSelectionError, match="index 0 is video, not audio"):
        select_audio_stream(media_info(video, audio), index=0)


def test_select_audio_stream_rejects_missing_audio_and_unknown_index() -> None:
    video_only = media_info(MediaStream(index=0, kind="video", codec="hevc"))

    with pytest.raises(AudioSelectionError, match="has no audio streams"):
        select_audio_stream(video_only)
    with pytest.raises(AudioSelectionError, match="index 3 does not exist"):
        select_audio_stream(video_only, index=3)


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
                "tags": {"language": "eng", "name": "English"},
                "side_data_list": [{"side_data_type": "Dolby object audio metadata"}],
            },
            {"index": 2, "codec_type": "subtitle", "codec_name": "subrip"},
        ],
        "chapters": [
            {
                "start_time": "0.000000",
                "end_time": "12.500000",
                "tags": {"title": "Opening"},
            }
        ],
        "format": {
            "format_name": "matroska,webm",
            "duration": "3217.952",
            "size": "1994237191",
            "bit_rate": "4957779",
            "tags": {"title": "Episode"},
        },
    }
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffprobe")

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)

    info = probe_media(media)

    assert info.path == media.resolve()
    assert info.container == "matroska,webm"
    assert info.duration == pytest.approx(3217.952)
    assert info.size == 1994237191
    assert len(info.streams) == 3
    assert info.streams[0].frame_rate == pytest.approx(24000 / 1001)
    assert info.streams[0].dolby_vision == "Dolby Vision P8.1"
    assert info.streams[1].language == "eng"
    assert info.streams[1].title == "English"
    assert info.title == "Episode"
    assert info.chapters == (MediaChapter(0, 12.5, "Opening"),)
    assert info.streams[1].channels == 6
    assert "Dolby object audio metadata" in info.streams[1].audio_side_data[0]
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
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: None)

    with pytest.raises(MediaProbeError, match="ffprobe was not found in PATH"):
        probe_media(media)


def test_probe_media_reports_ffprobe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "broken.mkv"
    media.touch()
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffprobe")
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "Invalid data"
        ),
    )

    with pytest.raises(MediaProbeError, match="Invalid data"):
        probe_media(media)


def test_probe_media_rejects_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "broken.mkv"
    media.touch()
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: "/bin/ffprobe")
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "not json", ""),
    )

    with pytest.raises(MediaProbeError, match="ffprobe returned invalid JSON"):
        probe_media(media)
