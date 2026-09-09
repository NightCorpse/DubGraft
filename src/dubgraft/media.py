"""Media metadata probing through FFprobe."""

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dubgraft.config import ANALYSIS_SAMPLE_RATE
from dubgraft.execution import (
    CommandExitError,
    CommandLaunchError,
    ExecutableNotFoundError,
    resolve_executable,
    run_capture,
)


class MediaProbeError(RuntimeError):
    """Raised when media metadata cannot be read safely."""


class FFmpegError(RuntimeError):
    """Raised when FFmpeg is unavailable or cannot be executed safely."""


class AudioSelectionError(ValueError):
    """Raised when an audio stream cannot be selected unambiguously."""

    def __init__(
        self, message: str, candidates: tuple["MediaStream", ...] = ()
    ) -> None:
        super().__init__(message)
        self.candidates = candidates


@dataclass(frozen=True, slots=True)
class MediaStream:
    index: int
    kind: str
    codec: str
    profile: str | None = None
    duration: float | None = None
    bit_rate: int | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    pixel_format: str | None = None
    color_space: str | None = None
    color_range: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    chroma_location: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    language: str | None = None
    title: str | None = None
    default: bool = False
    forced: bool = False
    hearing_impaired: bool = False
    attached_picture: bool = False
    dolby_vision: str | None = None
    start_time: float | None = None
    dispositions: tuple[str, ...] = ()
    video_side_data: tuple[str, ...] = ()
    audio_side_data: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MediaChapter:
    start_time: float
    end_time: float
    title: str | None = None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    container: str
    duration: float | None
    size: int | None
    bit_rate: int | None
    streams: tuple[MediaStream, ...]
    title: str | None = None
    chapters: tuple[MediaChapter, ...] = ()
    added_audio_stream_index: int | None = None


@dataclass(frozen=True, slots=True)
class FFmpegInfo:
    executable: Path
    version: str


def _ffmpeg_executable() -> str:
    try:
        return resolve_executable("ffmpeg")
    except ExecutableNotFoundError as error:
        raise FFmpegError(str(error)) from error


def validate_ffmpeg() -> FFmpegInfo:
    """Locate FFmpeg and verify that the executable responds successfully."""
    ffmpeg = _ffmpeg_executable()

    command = [ffmpeg, "-version"]
    try:
        result = run_capture(command, timeout=10)
    except CommandExitError as error:
        raise FFmpegError(str(error)) from error
    except CommandLaunchError as error:
        raise FFmpegError(f"could not execute ffmpeg: {error}") from error
    first_line = result.stdout.splitlines()
    if not first_line:
        raise FFmpegError("ffmpeg returned no version information")
    return FFmpegInfo(executable=Path(ffmpeg), version=first_line[0])


def extract_audio_window(
    path: Path, stream_index: int, start: float, duration: float
) -> NDArray[np.float32]:
    """Decode one selected audio window to normalized mono PCM in memory."""
    if stream_index < 0:
        raise ValueError("audio stream index must be non-negative")
    if not math.isfinite(start) or start < 0:
        raise ValueError("audio window start must be a non-negative finite number")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("audio window duration must be a positive finite number")

    media_path = path.expanduser().resolve()
    if not media_path.is_file():
        raise FFmpegError(f"Media file does not exist: {media_path}")
    ffmpeg = _ffmpeg_executable()
    command = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(media_path),
        "-map",
        f"0:{stream_index}",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        result = run_capture(command, timeout=60, text=False)
    except CommandExitError as error:
        raise FFmpegError(str(error)) from error
    except CommandLaunchError as error:
        raise FFmpegError(f"could not extract audio with ffmpeg: {error}") from error
    if not result.stdout:
        return np.array([], dtype=np.float32)
    if len(result.stdout) % np.dtype(np.int16).itemsize:
        raise FFmpegError("ffmpeg returned incomplete PCM audio data")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def select_audio_stream(info: MediaInfo, index: int | None = None) -> MediaStream:
    """Select one audio stream using its global FFprobe stream index."""
    candidates = tuple(stream for stream in info.streams if stream.kind == "audio")
    if index is None:
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise AudioSelectionError("media has no audio streams")
        raise AudioSelectionError(
            "media has multiple audio streams; select one by index", candidates
        )

    for stream in info.streams:
        if stream.index != index:
            continue
        if stream.kind != "audio":
            raise AudioSelectionError(
                f"stream index {index} is {stream.kind}, not audio", candidates
            )
        return stream
    raise AudioSelectionError(f"stream index {index} does not exist", candidates)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _frame_rate(value: Any) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None


def _dolby_vision(side_data: Any) -> str | None:
    if not isinstance(side_data, list):
        return None
    for entry in side_data:
        if not isinstance(entry, dict):
            continue
        if entry.get("side_data_type") != "DOVI configuration record":
            continue
        profile = _optional_int(entry.get("dv_profile"))
        compatibility = _optional_int(entry.get("dv_bl_signal_compatibility_id"))
        if profile is None:
            return "Dolby Vision"
        if compatibility is None:
            return f"Dolby Vision P{profile}"
        return f"Dolby Vision P{profile}.{compatibility}"
    return None


def _stream_side_data(raw: Any, kind: str, expected_kind: str) -> tuple[str, ...]:
    if kind != expected_kind or not isinstance(raw, list):
        return ()
    return tuple(
        json.dumps(entry, sort_keys=True, separators=(",", ":"))
        for entry in raw
        if isinstance(entry, dict)
    )


def _parse_stream(raw: Any) -> MediaStream:
    if not isinstance(raw, dict):
        raise MediaProbeError("ffprobe returned invalid stream data")
    try:
        index = int(raw["index"])
        kind = str(raw["codec_type"])
    except (KeyError, TypeError, ValueError) as error:
        raise MediaProbeError("ffprobe returned incomplete stream data") from error

    tags = raw.get("tags") if isinstance(raw.get("tags"), dict) else {}
    disposition = (
        raw.get("disposition") if isinstance(raw.get("disposition"), dict) else {}
    )
    dispositions = tuple(
        sorted(name for name, enabled in disposition.items() if bool(enabled))
    )
    side_data = raw.get("side_data_list")
    return MediaStream(
        index=index,
        kind=kind,
        codec=str(raw.get("codec_name") or "unknown"),
        profile=str(raw["profile"]) if raw.get("profile") else None,
        duration=_optional_float(raw.get("duration")),
        bit_rate=_optional_int(raw.get("bit_rate")),
        width=_optional_int(raw.get("width")),
        height=_optional_int(raw.get("height")),
        frame_rate=_frame_rate(raw.get("avg_frame_rate")),
        pixel_format=raw.get("pix_fmt"),
        color_space=raw.get("color_space"),
        color_range=raw.get("color_range"),
        color_transfer=raw.get("color_transfer"),
        color_primaries=raw.get("color_primaries"),
        chroma_location=raw.get("chroma_location"),
        sample_rate=_optional_int(raw.get("sample_rate")),
        channels=_optional_int(raw.get("channels")),
        channel_layout=raw.get("channel_layout"),
        language=tags.get("language"),
        title=tags.get("title") or tags.get("name"),
        default=bool(disposition.get("default")),
        forced=bool(disposition.get("forced")),
        hearing_impaired=bool(disposition.get("hearing_impaired")),
        attached_picture=bool(disposition.get("attached_pic")),
        dolby_vision=_dolby_vision(side_data),
        start_time=_optional_float(raw.get("start_time")),
        dispositions=dispositions,
        video_side_data=_stream_side_data(side_data, kind, "video"),
        audio_side_data=_stream_side_data(side_data, kind, "audio"),
    )


def _parse_chapter(raw: Any) -> MediaChapter:
    if not isinstance(raw, dict):
        raise MediaProbeError("ffprobe returned invalid chapter data")
    start_time = _optional_float(raw.get("start_time"))
    end_time = _optional_float(raw.get("end_time"))
    if start_time is None or end_time is None:
        raise MediaProbeError("ffprobe returned incomplete chapter data")
    tags = raw.get("tags") if isinstance(raw.get("tags"), dict) else {}
    return MediaChapter(start_time, end_time, tags.get("title"))


def probe_media(path: Path) -> MediaInfo:
    media_path = path.expanduser().resolve()
    if not media_path.exists():
        raise MediaProbeError(f"Media file does not exist: {media_path}")
    if not media_path.is_file():
        raise MediaProbeError(f"Media path is not a file: {media_path}")

    try:
        ffprobe = resolve_executable("ffprobe")
    except ExecutableNotFoundError as error:
        raise MediaProbeError(str(error)) from error

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        result = run_capture(command, timeout=60)
    except CommandExitError as error:
        raise MediaProbeError(str(error)) from error
    except CommandLaunchError as error:
        raise MediaProbeError(f"could not execute ffprobe: {error}") from error

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise MediaProbeError("ffprobe returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise MediaProbeError("ffprobe returned invalid media data")

    raw_format = payload.get("format")
    raw_streams = payload.get("streams")
    raw_chapters = payload.get("chapters", [])
    if (
        not isinstance(raw_format, dict)
        or not isinstance(raw_streams, list)
        or not isinstance(raw_chapters, list)
    ):
        raise MediaProbeError("ffprobe returned incomplete media data")
    format_tags = (
        raw_format.get("tags") if isinstance(raw_format.get("tags"), dict) else {}
    )

    return MediaInfo(
        path=media_path,
        container=str(raw_format.get("format_name") or "unknown"),
        duration=_optional_float(raw_format.get("duration")),
        size=_optional_int(raw_format.get("size")),
        bit_rate=_optional_int(raw_format.get("bit_rate")),
        streams=tuple(_parse_stream(stream) for stream in raw_streams),
        title=format_tags.get("title"),
        chapters=tuple(_parse_chapter(chapter) for chapter in raw_chapters),
    )
