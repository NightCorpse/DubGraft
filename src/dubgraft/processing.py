"""Processing orchestration and output muxing."""

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dubgraft.config import MatchingConfig, ProcessingConfig, TimelineConfig
from dubgraft.matching import (
    MatchingResult,
    TimelineAnalysis,
    TimelineKind,
    match_audio_streams,
)
from dubgraft.media import (
    AudioSelectionError,
    FFmpegError,
    MediaInfo,
    MediaProbeError,
    MediaStream,
    probe_media,
    select_audio_stream,
)


class ProcessingError(RuntimeError):
    """Raised when a valid processing request cannot produce an output."""


class InconclusiveTimelineError(ProcessingError):
    """Raised when matching cannot determine a safe temporal relationship."""

    def __init__(
        self,
        analysis: TimelineAnalysis,
        result: MatchingResult | None = None,
    ) -> None:
        self.result = result
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


def _publish_output(temporary_output: Path, config: ProcessingConfig) -> None:
    if config.overwrite:
        temporary_output.replace(config.output)
        return
    try:
        os.link(temporary_output, config.output)
    except FileExistsError as error:
        raise ProcessingError(
            f"Output already exists: {config.output}; use --overwrite to replace it"
        ) from error


def _target_duration(target_info: MediaInfo) -> float:
    duration = target_info.duration
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise ProcessingError("Target duration is unavailable")
    return duration


def _source_audio_duration(
    source_info: MediaInfo, source_audio: MediaStream
) -> float | None:
    for duration in (source_audio.duration, source_info.duration):
        if duration is not None and math.isfinite(duration) and duration > 0:
            return duration
    return None


def _audio_metadata_arguments(
    target_info: MediaInfo, source_audio: MediaStream
) -> list[str]:
    audio_index = sum(stream.kind == "audio" for stream in target_info.streams)
    arguments = []
    if source_audio.language:
        arguments.extend(
            [f"-metadata:s:a:{audio_index}", f"language={source_audio.language}"]
        )
    if source_audio.title:
        arguments.extend(
            [f"-metadata:s:a:{audio_index}", f"title={source_audio.title}"]
        )
    return arguments


def _validate_reconstructed_audio(path: Path, source_audio: MediaStream) -> None:
    try:
        reconstructed = select_audio_stream(probe_media(path))
    except (MediaProbeError, AudioSelectionError) as error:
        raise ProcessingError(
            f"could not validate reconstructed audio: {error}"
        ) from error

    for name, source_value, output_value in (
        ("channel count", source_audio.channels, reconstructed.channels),
        ("channel layout", source_audio.channel_layout, reconstructed.channel_layout),
        ("sample rate", source_audio.sample_rate, reconstructed.sample_rate),
    ):
        if source_value is not None and source_value != output_value:
            raise ProcessingError(
                f"reconstructed E-AC-3 changed {name} from {source_value} to "
                f"{output_value}; no output was created to avoid an implicit "
                "audio conversion"
            )


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
    target_duration = _target_duration(target_info)
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
                    str(target_duration),
                    "-avoid_negative_ts",
                    "disabled",
                ]
            )
            command.extend(_audio_metadata_arguments(target_info, source_audio))
            command.append(str(temporary_output))
            _run_ffmpeg(command, "mux")
            if not temporary_output.is_file():
                raise FFmpegError("ffmpeg did not create the output file")
            _publish_output(temporary_output, config)
    except OSError as error:
        raise ProcessingError(f"could not publish Output: {error}") from error


def mux_drift_audio(
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
    analysis: TimelineAnalysis,
) -> None:
    """Continuously retime and append Source audio to the Target timeline."""
    target_duration = _target_duration(target_info)
    slope = analysis.slope
    intercept = analysis.intercept
    if (
        slope is None
        or intercept is None
        or not math.isfinite(slope)
        or not math.isfinite(intercept)
        or slope <= 0
    ):
        raise ProcessingError("drift timeline has an invalid linear model")
    atmos_description = " ".join(
        value for value in (source_audio.profile, source_audio.title) if value
    )
    if source_audio.codec.casefold() == "truehd":
        raise ProcessingError(
            "selected Source audio uses TrueHD; the E-AC-3 drift path cannot "
            "preserve its lossless encoding or reliably preserve possible Atmos "
            "metadata; no output was created"
        )
    if "atmos" in atmos_description.casefold():
        raise ProcessingError(
            "selected Source audio contains Dolby Atmos metadata; drift requires "
            "re-encoding and would discard Atmos; no output was created"
        )
    if source_audio.channels is None:
        raise ProcessingError(
            "could not determine the selected Source audio channel count; no output "
            "was created to avoid an implicit downmix"
        )
    if source_audio.channel_layout is None:
        raise ProcessingError(
            "could not determine the selected Source audio channel layout; no output "
            "was created to avoid an implicit audio conversion"
        )
    if source_audio.sample_rate is None:
        raise ProcessingError(
            "could not determine the selected Source audio sample rate; no output "
            "was created to avoid an implicit audio conversion"
        )
    if source_audio.channels > 6:
        raise ProcessingError(
            f"selected Source audio has {source_audio.channels} channels; "
            "E-AC-3 drift output supports at most 6; no output was created "
            "to avoid downmix"
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegError("ffmpeg was not found in PATH")

    trim_start = max(intercept, 0.0)
    delay = max(-intercept / slope, 0.0)
    filters = [
        "asetpts=PTS-STARTPTS",
        f"atrim=start={trim_start:.9f}",
        "asetpts=PTS-STARTPTS",
        f"atempo={slope:.12f}",
    ]
    if delay > 0:
        filters.append(f"adelay={delay * 1000:.9f}:all=1")
    filters.extend(["apad", f"atrim=duration={target_duration:.9f}"])
    filter_graph = f"[0:{source_audio.index}]" + ",".join(filters) + "[dubbed]"

    try:
        with tempfile.TemporaryDirectory(
            prefix=".dubgraft-", dir=config.output.parent
        ) as temporary_directory:
            reconstructed_audio = Path(temporary_directory) / "reconstructed-audio.mka"
            reconstruction_command = [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                str(config.source),
                "-filter_complex",
                filter_graph,
                "-map",
                "[dubbed]",
                "-c:a",
                "eac3",
                "-b:a",
                "640k",
                "-f",
                "matroska",
                str(reconstructed_audio),
            ]
            _run_ffmpeg(reconstruction_command, "drift reconstruction")
            if not reconstructed_audio.is_file():
                raise FFmpegError("ffmpeg did not create the reconstructed audio")
            _validate_reconstructed_audio(reconstructed_audio, source_audio)

            temporary_output = Path(temporary_directory) / config.output.name
            command = [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                str(config.target),
                "-i",
                str(reconstructed_audio),
                "-map",
                "0",
                "-map",
                "1:0",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                "-c",
                "copy",
                "-copy_unknown",
                "-max_interleave_delta",
                "0",
                "-t",
                str(target_duration),
                "-avoid_negative_ts",
                "disabled",
            ]
            command.extend(_audio_metadata_arguments(target_info, source_audio))
            command.append(str(temporary_output))
            _run_ffmpeg(command, "mux")
            if not temporary_output.is_file():
                raise FFmpegError("ffmpeg did not create the output file")
            _publish_output(temporary_output, config)
    except OSError as error:
        raise ProcessingError(f"could not publish Output: {error}") from error


def process_media(
    config: ProcessingConfig,
    source_info: MediaInfo,
    target_info: MediaInfo,
    source_audio: MediaStream,
    target_audio: MediaStream,
    matching_config: MatchingConfig = MatchingConfig(),
    timeline_config: TimelineConfig = TimelineConfig(),
) -> MatchingResult:
    """Analyze selected streams and produce a synchronized output."""
    target_duration = _target_duration(target_info)

    result = match_audio_streams(
        config.source,
        source_audio.index,
        config.target,
        target_audio.index,
        target_duration,
        matching_config,
        timeline_config,
        source_duration=_source_audio_duration(source_info, source_audio),
    )
    if result.timeline.kind is TimelineKind.INCONCLUSIVE:
        raise InconclusiveTimelineError(result.timeline, result)
    if result.timeline.kind is TimelineKind.DRIFT:
        mux_drift_audio(config, target_info, source_audio, result.timeline)
        return result

    offset = 0.0
    if result.timeline.kind is TimelineKind.STATIC:
        if result.timeline.median_offset is None:
            raise ProcessingError("static timeline has no offset")
        offset = result.timeline.median_offset
    mux_source_audio(config, target_info, source_audio, offset=offset)
    return result
