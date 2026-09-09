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
    arguments.extend(
        [
            f"-disposition:a:{audio_index}",
            "+".join(source_audio.dispositions) or "0",
        ]
    )
    return arguments


def _target_stream_metadata_arguments(target_info: MediaInfo) -> list[str]:
    arguments = []
    video_index = 0
    for position, stream in enumerate(target_info.streams):
        if stream.language:
            arguments.extend([f"-metadata:s:{position}", f"language={stream.language}"])
        if stream.title:
            arguments.extend([f"-metadata:s:{position}", f"title={stream.title}"])
        arguments.extend(
            [f"-disposition:{position}", "+".join(stream.dispositions) or "0"]
        )
        if stream.kind == "video":
            for option, value in (
                ("color_range", stream.color_range),
                ("colorspace", stream.color_space),
                ("color_trc", stream.color_transfer),
                ("color_primaries", stream.color_primaries),
                ("chroma_sample_location", stream.chroma_location),
            ):
                if value:
                    arguments.extend([f"-{option}:v:{video_index}", value])
            video_index += 1
    return arguments


def _container_preservation_arguments(output: Path, target_info: MediaInfo) -> list[str]:
    if output.suffix.casefold() in {".mp4", ".m4v", ".mov"} and any(
        stream.dolby_vision for stream in target_info.streams
    ):
        return ["-strict", "unofficial"]
    return []


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
    audio_start_time: float = 0.0,
    audio_duration: float | None = None,
) -> MediaInfo:
    def fail(detail: str) -> Never:
        raise ProcessingError(
            f"generated Output failed validation: {detail}; Output was not published"
        )

    try:
        output_info = probe_media(path)
    except MediaProbeError as error:
        fail(str(error))

    minimum_stream_count = len(target_info.streams) + 1
    if len(output_info.streams) < minimum_stream_count:
        fail(
            f"expected at least {minimum_stream_count} streams, found "
            f"{len(output_info.streams)}"
        )

    target_audio_count = sum(stream.kind == "audio" for stream in target_info.streams)
    output_audio_positions = [
        position
        for position, stream in enumerate(output_info.streams)
        if stream.kind == "audio"
    ]
    if len(output_audio_positions) != target_audio_count + 1:
        fail(
            f"expected {target_audio_count + 1} audio streams, found "
            f"{len(output_audio_positions)}"
        )
    language, title = _audio_metadata(config, source_audio)
    matching_added_positions = [
        position
        for position in output_audio_positions
        if all(
            getattr(output_info.streams[position], name) == expected
            for name, expected in (
                ("codec", audio_codec),
                ("channels", source_audio.channels),
                ("channel_layout", source_audio.channel_layout),
                ("sample_rate", source_audio.sample_rate),
            )
            if expected is not None
        )
    ]

    expected_position = output_audio_positions[target_audio_count]

    def identification_score(position: int) -> tuple[int, int, float]:
        stream = output_info.streams[position]
        metadata_error = int(stream.title != title)
        if language is None:
            metadata_error += int(stream.language not in {None, "und"})
        else:
            metadata_error += int(stream.language != language)
        timing_error = abs((stream.start_time or 0.0) - audio_start_time)
        if audio_duration is not None and stream.duration is not None:
            timing_error += abs(stream.duration - audio_duration)
        return metadata_error, abs(position - expected_position), timing_error

    added_audio_position = min(
        matching_added_positions or [output_audio_positions[target_audio_count]],
        key=identification_score,
    )
    added_audio = output_info.streams[added_audio_position]
    output_without_added = (
        output_info.streams[:added_audio_position]
        + output_info.streams[added_audio_position + 1 :]
    )
    preserved_streams = []
    used_positions: set[int] = set()
    kind_offsets: dict[str, int] = {}
    for target_stream in target_info.streams:
        start = kind_offsets.get(target_stream.kind, 0)
        match = next(
            (
                (position, stream)
                for position, stream in enumerate(output_without_added[start:], start)
                if stream.kind == target_stream.kind
            ),
            None,
        )
        if match is None:
            fail(f"Target stream {target_stream.index} was not preserved")
        position, stream = match
        kind_offsets[target_stream.kind] = position + 1
        used_positions.add(position)
        preserved_streams.append(stream)
    extra_streams = tuple(
        stream
        for position, stream in enumerate(output_without_added)
        if position not in used_positions
    )
    if extra_streams and not (
        target_info.chapters
        and path.suffix.casefold() in {".mp4", ".m4v", ".mov"}
        and all(stream.kind == "data" for stream in extra_streams)
    ):
        fail(f"Output contains {len(extra_streams)} unexpected streams")

    for position, (target_stream, output_stream) in enumerate(
        zip(target_info.streams, preserved_streams)
    ):
        properties = ["kind", "codec", "language", "title"]
        if target_stream.kind == "video":
            properties.extend(
                [
                    "profile",
                    "width",
                    "height",
                    "pixel_format",
                    "color_space",
                    "color_range",
                    "color_transfer",
                    "color_primaries",
                    "chroma_location",
                    "dolby_vision",
                    "video_side_data",
                ]
            )
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
        comparable_dispositions = output_stream.dispositions
        if (
            path.suffix.casefold() in {".mp4", ".m4v", ".mov"}
            and "attached_pic" in target_stream.dispositions
            and "timed_thumbnails" in comparable_dispositions
        ):
            comparable_dispositions = tuple(
                value
                for value in comparable_dispositions
                if value != "timed_thumbnails"
            )
        dispositions_match = comparable_dispositions == target_stream.dispositions
        if (
            not dispositions_match
            and path.suffix.casefold() in {".mp4", ".m4v", ".mov"}
            and "default" not in target_stream.dispositions
            and not any(
                stream.kind == target_stream.kind
                and "default" in stream.dispositions
                for stream in target_info.streams
            )
        ):
            dispositions_match = tuple(
                value for value in comparable_dispositions if value != "default"
            ) == target_stream.dispositions
        if not dispositions_match:
            fail(
                f"Target stream {position} dispositions expected "
                f"{target_stream.dispositions!r}, found {output_stream.dispositions!r}"
            )

    if target_info.title is not None and output_info.title != target_info.title:
        fail(
            f"Target title expected {target_info.title!r}, found {output_info.title!r}"
        )
    if len(output_info.chapters) != len(target_info.chapters):
        fail(
            f"expected {len(target_info.chapters)} chapters, found "
            f"{len(output_info.chapters)}"
        )
    for position, (target_chapter, output_chapter) in enumerate(
        zip(target_info.chapters, output_info.chapters)
    ):
        if (
            abs(output_chapter.start_time - target_chapter.start_time) > 0.05
            or abs(output_chapter.end_time - target_chapter.end_time) > 0.05
            or output_chapter.title != target_chapter.title
        ):
            fail(f"Target chapter {position} changed during muxing")

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
    added_dispositions_match = added_audio.dispositions == source_audio.dispositions
    if (
        not added_dispositions_match
        and path.suffix.casefold() in {".mp4", ".m4v", ".mov"}
        and "default" not in source_audio.dispositions
    ):
        added_dispositions_match = tuple(
            value for value in added_audio.dispositions if value != "default"
        ) == source_audio.dispositions
    if not added_dispositions_match:
        fail(
            f"added audio dispositions expected {source_audio.dispositions!r}, "
            f"found {added_audio.dispositions!r}"
        )

    actual_start_time = added_audio.start_time or 0.0
    if abs(actual_start_time - audio_start_time) > 0.1:
        fail(
            f"added audio start expected {audio_start_time:.3f}s, found "
            f"{actual_start_time:.3f}s"
        )
    if (
        audio_duration is not None
        and added_audio.duration is not None
        and abs(added_audio.duration - audio_duration) > 0.1
    ):
        fail(
            f"added audio duration expected {audio_duration:.3f}s, found "
            f"{added_audio.duration:.3f}s"
        )

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
    return replace(output_info, added_audio_stream_index=added_audio.index)


def validate_mux_compatibility(
    config: ProcessingConfig, target_info: MediaInfo
) -> None:
    """Check that the requested container can copy every Target stream."""
    output = _output_path(config)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegError("ffmpeg was not found in PATH")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".dubgraft-preflight-", dir=output.parent
        ) as temporary_directory:
            temporary_output = Path(temporary_directory) / f"preflight{output.suffix}"
            command = [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                str(config.target),
                "-map",
                "0",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                "-c",
                "copy",
                "-copy_unknown",
                "-t",
                "0.1",
            ]
            command.extend(_target_stream_metadata_arguments(target_info))
            command.extend(
                _container_preservation_arguments(temporary_output, target_info)
            )
            command.append(str(temporary_output))
            try:
                _run_ffmpeg(command, "Output compatibility check")
            except FFmpegError as error:
                target_extension = target_info.path.suffix
                suggestion = (
                    f"; use an Output with the Target extension ({target_extension})"
                    if target_extension
                    and target_extension.casefold() != output.suffix.casefold()
                    else ""
                )
                raise ProcessingError(
                    f"Output container cannot preserve every Target stream{suggestion}: "
                    f"{error}"
                ) from error
    except OSError as error:
        raise ProcessingError(f"could not check Output compatibility: {error}") from error


def validate_render_compatibility(
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
    timeline_kind: TimelineKind,
) -> None:
    """Check the selected audio strategy against the requested container."""
    output = _output_path(config)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegError("ffmpeg was not found in PATH")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".dubgraft-audio-preflight-", dir=output.parent
        ) as temporary_directory:
            temporary_output = Path(temporary_directory) / f"preflight{output.suffix}"
            command = [ffmpeg, "-v", "error", "-y", "-i", str(config.target)]
            if timeline_kind is TimelineKind.DRIFT:
                command.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=48000:cl=stereo:d=0.1",
                    ]
                )
                source_stream = "1:0"
            else:
                command.extend(["-i", str(config.source)])
                source_stream = f"1:{source_audio.index}"
            command.extend(
                [
                    "-map",
                    "0",
                    "-map",
                    source_stream,
                    "-map_metadata",
                    "0",
                    "-map_chapters",
                    "0",
                    "-c",
                    "copy",
                    "-copy_unknown",
                    "-t",
                    "0.1",
                ]
            )
            if timeline_kind is TimelineKind.DRIFT:
                audio_index = sum(
                    stream.kind == "audio" for stream in target_info.streams
                )
                command.extend([f"-c:a:{audio_index}", "eac3", "-b:a", "640k"])
            command.extend(_target_stream_metadata_arguments(target_info))
            command.extend(_audio_metadata_arguments(config, target_info, source_audio))
            command.extend(
                _container_preservation_arguments(temporary_output, target_info)
            )
            command.append(str(temporary_output))
            try:
                _run_ffmpeg(command, "audio compatibility check")
            except FFmpegError as error:
                suggestion = (
                    "; try Matroska (.mkv) if it can preserve all Target streams"
                    if output.suffix.casefold() != ".mkv"
                    else ""
                )
                raise ProcessingError(
                    f"selected audio strategy is incompatible with the Output "
                    f"container{suggestion}: {error}"
                ) from error
    except OSError as error:
        raise ProcessingError(f"could not check audio compatibility: {error}") from error


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
            command.extend(_target_stream_metadata_arguments(target_info))
            command.extend(_audio_metadata_arguments(config, target_info, source_audio))
            command.extend(
                _container_preservation_arguments(temporary_output, target_info)
            )
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
                audio_start_time=max(-offset, 0.0),
                audio_duration=(
                    max(
                        0.0,
                        min(
                            target_duration - max(-offset, 0.0),
                            source_duration - max(offset, 0.0),
                        ),
                    )
                    if source_duration is not None
                    else None
                ),
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
            command.extend(_target_stream_metadata_arguments(target_info))
            command.extend(_audio_metadata_arguments(config, target_info, source_audio))
            command.extend(
                _container_preservation_arguments(temporary_output, target_info)
            )
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
                audio_start_time=0.0,
                audio_duration=target_duration,
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
