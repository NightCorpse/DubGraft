"""Processing orchestration and output muxing."""

import logging
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Never

from dubgraft.config import (
    ConfigurationError,
    MatchingConfig,
    ProcessingConfig,
    TimelineConfig,
    normalize_track_name,
)
from dubgraft.languages import LanguageCodeError, normalize_language_code
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


logger = logging.getLogger(__name__)
RenderProgress = Callable[[str, float, float], None]


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


def _output_path(config: ProcessingConfig) -> Path:
    if config.output is None:
        raise ProcessingError("Output is required for media processing")
    return config.output


def _progress_time(values: dict[str, str]) -> float | None:
    for key in ("out_time_us", "out_time_ms"):
        try:
            seconds = int(values[key]) / 1_000_000
        except (KeyError, ValueError):
            continue
        if math.isfinite(seconds):
            return seconds

    try:
        hours, minutes, seconds_text = values["out_time"].split(":", 2)
        seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_text)
    except (KeyError, ValueError):
        return None
    return seconds if math.isfinite(seconds) else None


def _iter_progress_times(lines: Iterable[str]) -> Iterator[float]:
    values: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.strip().partition("=")
        if not separator:
            continue
        values[key] = value
        if key == "progress":
            seconds = _progress_time(values)
            if seconds is not None:
                yield seconds
            values.clear()


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait()
        except OSError:
            pass
    except OSError:
        pass


def _run_ffmpeg(
    command: list[str],
    operation: str,
    *,
    expected_duration: float | None = None,
    progress: RenderProgress | None = None,
) -> None:
    if progress is not None and expected_duration is not None:
        _run_ffmpeg_with_progress(command, operation, expected_duration, progress)
        return

    logger.debug("Running command: %s", shlex.join(command))
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


def _run_ffmpeg_with_progress(
    command: list[str],
    operation: str,
    expected_duration: float,
    progress: RenderProgress,
) -> None:
    progress_command = [
        command[0],
        "-nostats",
        "-progress",
        "pipe:1",
        *command[1:],
    ]
    logger.debug("Running command: %s", shlex.join(progress_command))
    try:
        with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        ) as diagnostics:
            process = subprocess.Popen(
                progress_command,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            completed = 0.0
            try:
                for seconds in _iter_progress_times(process.stdout):
                    current = min(max(seconds, completed, 0.0), expected_duration * 0.99)
                    if current > completed:
                        completed = current
                        progress(operation, completed, expected_duration)
                return_code = process.wait()
            except BaseException:
                _stop_process(process)
                raise
            finally:
                process.stdout.close()

            diagnostics.seek(0)
            detail = diagnostics.read().strip()
    except OSError as error:
        raise FFmpegError(f"could not execute ffmpeg {operation}: {error}") from error

    if return_code != 0:
        raise FFmpegError(detail or f"ffmpeg exited with code {return_code}")
    progress(operation, expected_duration, expected_duration)


def _publish_output(temporary_output: Path, config: ProcessingConfig) -> None:
    output = _output_path(config)
    if config.overwrite:
        temporary_output.replace(output)
        return
    try:
        os.link(temporary_output, output)
    except FileExistsError as error:
        raise ProcessingError(
            f"Output already exists: {output}; use --overwrite to replace it"
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
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
) -> list[str]:
    audio_index = sum(stream.kind == "audio" for stream in target_info.streams)
    arguments = []
    language, title = _audio_metadata(config, source_audio)
    if language:
        arguments.extend(
            [f"-metadata:s:a:{audio_index}", f"language={language}"]
        )
    if title:
        arguments.extend(
            [f"-metadata:s:a:{audio_index}", f"title={title}"]
        )
    return arguments


def _audio_metadata(
    config: ProcessingConfig, source_audio: MediaStream
) -> tuple[str | None, str | None]:
    if config.language is None:
        language = source_audio.language
    else:
        try:
            language = normalize_language_code(config.language)
        except LanguageCodeError as error:
            raise ProcessingError(str(error)) from error
    if config.track_name is None:
        title = source_audio.title
    else:
        try:
            title = normalize_track_name(config.track_name)
        except ConfigurationError as error:
            raise ProcessingError(str(error)) from error
    return language, title


def _validate_output(
    path: Path,
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
    *,
    audio_codec: str,
) -> MediaInfo:
    def fail(detail: str) -> Never:
        raise ProcessingError(
            f"generated Output failed validation: {detail}; Output was not published"
        )

    try:
        output_info = probe_media(path)
    except MediaProbeError as error:
        fail(str(error))

    expected_stream_count = len(target_info.streams) + 1
    if len(output_info.streams) != expected_stream_count:
        fail(
            f"expected {expected_stream_count} streams, found "
            f"{len(output_info.streams)}"
        )

    for position, (target_stream, output_stream) in enumerate(
        zip(target_info.streams, output_info.streams)
    ):
        properties = ["kind", "codec", "language", "title"]
        if target_stream.kind == "video":
            properties.extend(["width", "height", "pixel_format"])
        elif target_stream.kind == "audio":
            properties.extend(["channels", "channel_layout", "sample_rate"])
        for name in properties:
            expected = getattr(target_stream, name)
            actual = getattr(output_stream, name)
            if expected is not None and actual != expected:
                fail(
                    f"Target stream {position} {name} expected {expected!r}, "
                    f"found {actual!r}"
                )

    added_audio = output_info.streams[-1]
    if added_audio.kind != "audio":
        fail(f"added stream expected audio, found {added_audio.kind}")

    language, title = _audio_metadata(config, source_audio)
    for name, expected, actual in (
        ("codec", audio_codec, added_audio.codec),
        ("channel count", source_audio.channels, added_audio.channels),
        ("channel layout", source_audio.channel_layout, added_audio.channel_layout),
        ("sample rate", source_audio.sample_rate, added_audio.sample_rate),
        ("language", language, added_audio.language),
        ("title", title, added_audio.title),
    ):
        if expected is not None and actual != expected:
            fail(f"added audio {name} expected {expected!r}, found {actual!r}")

    target_duration = _target_duration(target_info)
    output_duration = output_info.duration
    if (
        output_duration is None
        or not math.isfinite(output_duration)
        or abs(output_duration - target_duration) > 0.1
    ):
        fail(
            f"duration expected {target_duration:.3f}s, found "
            f"{output_duration if output_duration is not None else 'unavailable'}"
        )
    return output_info


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
    source_duration: float | None = None,
    progress: RenderProgress | None = None,
) -> MediaInfo:
    """Copy the Target and append one Source audio stream at a static offset."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegError("ffmpeg was not found in PATH")
    target_duration = _target_duration(target_info)
    output = _output_path(config)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".dubgraft-", dir=output.parent
        ) as temporary_directory:
            temporary_output = Path(temporary_directory) / output.name
            source_path = config.source
            source_stream_index = source_audio.index
            if offset > 0:
                source_path = Path(temporary_directory) / "trimmed-source.mka"
                trim_duration = next(
                    (
                        duration
                        for duration in (source_audio.duration, source_duration)
                        if duration is not None
                        and math.isfinite(duration)
                        and duration > offset
                    ),
                    None,
                )
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
                    expected_duration=trim_duration,
                    progress=progress,
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
            command.extend(_audio_metadata_arguments(config, target_info, source_audio))
            command.append(str(temporary_output))
            _run_ffmpeg(
                command,
                "mux",
                expected_duration=target_duration,
                progress=progress,
            )
            if not temporary_output.is_file():
                raise FFmpegError("ffmpeg did not create the output file")
            output_info = _validate_output(
                temporary_output,
                config,
                target_info,
                source_audio,
                audio_codec=source_audio.codec,
            )
            _publish_output(temporary_output, config)
            return replace(output_info, path=output.expanduser().resolve())
    except OSError as error:
        raise ProcessingError(f"could not publish Output: {error}") from error


def mux_drift_audio(
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
    analysis: TimelineAnalysis,
    *,
    progress: RenderProgress | None = None,
) -> MediaInfo:
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

    output = _output_path(config)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".dubgraft-", dir=output.parent
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
            _run_ffmpeg(
                reconstruction_command,
                "audio reconstruction",
                expected_duration=target_duration,
                progress=progress,
            )
            if not reconstructed_audio.is_file():
                raise FFmpegError("ffmpeg did not create the reconstructed audio")
            _validate_reconstructed_audio(reconstructed_audio, source_audio)

            temporary_output = Path(temporary_directory) / output.name
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
            command.extend(_audio_metadata_arguments(config, target_info, source_audio))
            command.append(str(temporary_output))
            _run_ffmpeg(
                command,
                "mux",
                expected_duration=target_duration,
                progress=progress,
            )
            if not temporary_output.is_file():
                raise FFmpegError("ffmpeg did not create the output file")
            output_info = _validate_output(
                temporary_output,
                config,
                target_info,
                source_audio,
                audio_codec="eac3",
            )
            _publish_output(temporary_output, config)
            return replace(output_info, path=output.expanduser().resolve())
    except OSError as error:
        raise ProcessingError(f"could not publish Output: {error}") from error


def analyze_media(
    config: ProcessingConfig,
    source_info: MediaInfo,
    target_info: MediaInfo,
    source_audio: MediaStream,
    target_audio: MediaStream,
    matching_config: MatchingConfig | None = None,
    timeline_config: TimelineConfig | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> MatchingResult:
    """Analyze the temporal relationship between selected audio streams."""
    target_duration = _target_duration(target_info)
    matching_config = matching_config or config.matching_config
    timeline_config = timeline_config or config.timeline_config
    return match_audio_streams(
        config.source,
        source_audio.index,
        config.target,
        target_audio.index,
        target_duration,
        matching_config,
        timeline_config,
        source_duration=_source_audio_duration(source_info, source_audio),
        progress=progress,
    )


def render_media(
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
    result: MatchingResult,
    *,
    source_duration: float | None = None,
    progress: RenderProgress | None = None,
) -> MediaInfo:
    """Render a previously analyzed and conclusive matching result."""
    if result.timeline.kind is TimelineKind.INCONCLUSIVE:
        raise InconclusiveTimelineError(result.timeline, result)
    if result.timeline.kind is TimelineKind.DRIFT:
        return mux_drift_audio(
            config,
            target_info,
            source_audio,
            result.timeline,
            progress=progress,
        )

    offset = 0.0
    if result.timeline.kind is TimelineKind.STATIC:
        if result.timeline.median_offset is None:
            raise ProcessingError("static timeline has no offset")
        offset = result.timeline.median_offset
    return mux_source_audio(
        config,
        target_info,
        source_audio,
        offset=offset,
        source_duration=source_duration,
        progress=progress,
    )


def process_media(
    config: ProcessingConfig,
    source_info: MediaInfo,
    target_info: MediaInfo,
    source_audio: MediaStream,
    target_audio: MediaStream,
    matching_config: MatchingConfig | None = None,
    timeline_config: TimelineConfig | None = None,
) -> MatchingResult:
    """Analyze selected streams and produce a synchronized output."""
    _output_path(config)
    result = analyze_media(
        config,
        source_info,
        target_info,
        source_audio,
        target_audio,
        matching_config,
        timeline_config,
    )
    render_media(
        config,
        target_info,
        source_audio,
        result,
        source_duration=_source_audio_duration(source_info, source_audio),
    )
    return result
