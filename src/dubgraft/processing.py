"""Processing orchestration and output muxing."""

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dubgraft.config import MatchingConfig, ProcessingConfig, TimelineConfig
from dubgraft.matching import MatchingResult, TimelineAnalysis, TimelineKind, match_audio_streams
from dubgraft.media import FFmpegError, MediaInfo, MediaStream


class ProcessingError(RuntimeError):
    """Raised when a valid processing request cannot produce an output."""


class InconclusiveTimelineError(ProcessingError):
    """Raised when matching cannot determine a safe temporal relationship."""

    def __init__(self, analysis: TimelineAnalysis) -> None:
        self.analysis = analysis
        super().__init__(analysis.reason or "timeline analysis was inconclusive")


def _run_ffmpeg(command: list[str], operation: str) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        raise FFmpegError(f"could not execute ffmpeg {operation}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"ffmpeg exited with code {result.returncode}"
        raise FFmpegError(detail)


def mux_source_audio(
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
    *,
    offset: float,
) -> None:
    """Copy the Target and append one Source audio stream at a static offset."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegError("ffmpeg was not found in PATH")
    if (
        target_info.duration is None
        or not math.isfinite(target_info.duration)
        or target_info.duration <= 0
    ):
        raise ProcessingError("Target duration is unavailable")

    target_audio_count = sum(
        stream.kind == "audio" for stream in target_info.streams
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix=".dubgraft-", dir=config.output.parent
        ) as temporary_directory:
            temporary_output = Path(temporary_directory) / config.output.name
            source_path = config.source
            source_stream_index = source_audio.index
            if offset > 0:
                source_path = Path(temporary_directory) / "trimmed-source.mka"
                _run_ffmpeg(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(config.source),
                        "-ss",
                        str(offset),
                        "-map",
                        f"0:{source_audio.index}",
                        "-map_metadata",
                        "0",
                        "-c",
                        "copy",
                        "-avoid_negative_ts",
                        "disabled",
                        "-f",
                        "matroska",
                        str(source_path),
                    ],
                    "audio trim",
                )
                if not source_path.is_file():
                    raise FFmpegError("ffmpeg did not create the trimmed audio")
                source_stream_index = 0

            command = [ffmpeg, "-v", "error", "-y", "-i", str(config.target)]
            if offset < 0:
                command.extend(["-itsoffset", str(-offset)])
            command.extend(
                [
                    "-i",
                    str(source_path),
                    "-map",
                    "0",
                    "-map",
                    f"1:{source_stream_index}",
                    "-map_metadata",
                    "0",
                    "-map_chapters",
                    "0",
                    "-c",
                    "copy",
                    "-copy_unknown",
                    "-t",
                    str(target_info.duration),
                    "-avoid_negative_ts",
                    "disabled",
                ]
            )
            if source_audio.language:
                command.extend(
                    [
                        f"-metadata:s:a:{target_audio_count}",
                        f"language={source_audio.language}",
                    ]
                )
            if source_audio.title:
                command.extend(
                    [
                        f"-metadata:s:a:{target_audio_count}",
                        f"title={source_audio.title}",
                    ]
                )
            command.append(str(temporary_output))
            _run_ffmpeg(command, "mux")
            if not temporary_output.is_file():
                raise FFmpegError("ffmpeg did not create the output file")
            if config.overwrite:
                temporary_output.replace(config.output)
            else:
                try:
                    os.link(temporary_output, config.output)
                except FileExistsError as error:
                    raise ProcessingError(
                        f"Output already exists: {config.output}; use --overwrite to replace it"
                    ) from error
    except OSError as error:
        raise ProcessingError(f"could not publish Output: {error}") from error


def process_media(
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
    target_audio: MediaStream,
    matching_config: MatchingConfig = MatchingConfig(),
    timeline_config: TimelineConfig = TimelineConfig(),
) -> MatchingResult:
    """Analyze selected streams and produce direct or statically aligned output."""
    if (
        target_info.duration is None
        or not math.isfinite(target_info.duration)
        or target_info.duration <= 0
    ):
        raise ProcessingError("Target duration is unavailable")

    result = match_audio_streams(
        config.source,
        source_audio.index,
        config.target,
        target_audio.index,
        target_info.duration,
        matching_config,
        timeline_config,
    )
    if result.timeline.kind is TimelineKind.INCONCLUSIVE:
        raise InconclusiveTimelineError(result.timeline)
    if result.timeline.kind is TimelineKind.DRIFT:
        raise ProcessingError("timeline requires drift reconstruction")

    offset = 0.0
    if result.timeline.kind is TimelineKind.STATIC:
        if result.timeline.median_offset is None:
            raise ProcessingError("static timeline has no offset")
        offset = result.timeline.median_offset
    mux_source_audio(config, target_info, source_audio, offset=offset)
    return result
