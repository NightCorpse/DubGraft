"""Command-line entry point for DubGraft."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dubgraft import __version__
from dubgraft.config import (
    ConfigurationError,
    ProcessingConfig,
    validate_log_path,
    validate_processing_config,
)
from dubgraft.matching import MatchingResult, TimelineKind
from dubgraft.languages import LanguageCodeError, normalize_language_code
from dubgraft.media import (
    AudioSelectionError,
    FFmpegError,
    MediaInfo,
    MediaProbeError,
    MediaStream,
    probe_media,
    select_audio_stream,
    validate_ffmpeg,
)
from dubgraft.processing import (
    InconclusiveTimelineError,
    ProcessingError,
    analyze_media,
    render_media,
)
from dubgraft.report import (
    ReportError,
    build_report,
    format_human_report,
    write_json_report,
)
from dubgraft.runtime import RunOutput


def _format_offset(value: float) -> str:
    return f"{value:+.3f}s"


def format_processing_summary(
    result: MatchingResult,
    config: ProcessingConfig,
    target_info: MediaInfo,
    source_audio: MediaStream,
) -> str:
    """Format the processing decisions and completed output for the user."""
    timeline = result.timeline
    analysis = f"Analysis: {timeline.kind.value}"
    if timeline.used_duration_fallback:
        analysis += " (strict duration fallback)"
    lines = [
        analysis,
        f"Anchors: {len(result.anchors)}/{result.requested_anchor_count} | "
        f"coverage: {timeline.coverage:.1%}",
    ]
    if timeline.used_duration_fallback:
        duration_error = timeline.source_duration_error or 0.0
        lines.append(f"Fallback duration error: {duration_error:.3f}s")

    if timeline.kind is TimelineKind.DIRECT:
        offset = timeline.median_offset or 0.0
        lines.extend(
            [
                f"Offset: {_format_offset(offset)}",
                "Audio: copied without re-encoding",
            ]
        )
    elif timeline.kind is TimelineKind.STATIC:
        offset = timeline.median_offset or 0.0
        correction = (
            f"trimmed Source beginning by {offset:.3f}s"
            if offset > 0
            else f"delayed Source by {-offset:.3f}s"
        )
        lines.extend(
            [
                f"Offset: {_format_offset(offset)} | {correction}",
                "Audio: copied without re-encoding",
            ]
        )
    elif timeline.kind is TimelineKind.DRIFT:
        slope = timeline.slope or 1.0
        intercept = timeline.intercept or 0.0
        drift = timeline.drift_over_duration or 0.0
        lines.append(
            f"Model: slope {slope:.9f} | intercept {_format_offset(intercept)} | "
            f"total drift {_format_offset(drift)}"
        )
        layout = source_audio.channel_layout
        if not layout and source_audio.channels is not None:
            layout = f"{source_audio.channels} channels"
        audio = "Audio: E-AC-3 640 kb/s"
        if layout:
            audio += f", {layout}"
        lines.append(audio + ", re-encoded once")

    lines.extend(
        [
            f"Target streams: {len(target_info.streams)} preserved",
            f"Created: {config.output}",
        ]
    )
    return "\n".join(lines)


def format_analysis_summary(result: MatchingResult) -> str:
    """Format a concise result for analysis-only mode."""
    timeline = result.timeline
    analysis = f"Analysis: {timeline.kind.value}"
    if timeline.used_duration_fallback:
        analysis += " (strict duration fallback)"
    lines = [
        analysis,
        f"Anchors: {len(result.anchors)}/{result.requested_anchor_count} | "
        f"coverage: {timeline.coverage:.1%}",
    ]
    if timeline.median_offset is not None:
        lines.append(f"Median offset: {_format_offset(timeline.median_offset)}")
    if timeline.slope is not None and timeline.intercept is not None:
        drift = timeline.drift_over_duration or 0.0
        lines.append(
            f"Model: slope {timeline.slope:.9f} | "
            f"intercept {_format_offset(timeline.intercept)} | "
            f"total drift {_format_offset(drift)}"
        )
    lines.append("No output was created (--analyze-only).")
    return "\n".join(lines)


def format_inconclusive_error(error: InconclusiveTimelineError) -> str:
    """Format an inconclusive result with the evidence that was available."""
    result = error.result
    timeline = error.analysis
    if result is None:
        return f"inconclusive analysis: {error}\nNo output was created."
    lines = [
        f"inconclusive analysis: {error}",
        f"Anchors: {len(result.anchors)}/{result.requested_anchor_count} | "
        f"coverage: {timeline.coverage:.1%}",
    ]
    if timeline.source_duration_error is not None:
        lines.append(f"Source duration error: {timeline.source_duration_error:.3f}s")
    lines.append("No output was created.")
    return "\n".join(lines)


def _stream_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if index < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return index


def _language_code(value: str) -> str:
    try:
        return normalize_language_code(value)
    except LanguageCodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dubgraft",
        description="Align and graft dubbed audio across media releases using distributed audio anchors.",
        epilog="command:\n  inspect MEDIA [...]   show video, audio, and stream summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("positional_source", nargs="?", metavar="SOURCE")
    parser.add_argument("positional_target", nargs="?", metavar="TARGET")
    parser.add_argument("positional_output", nargs="?", metavar="OUTPUT")
    parser.add_argument("-s", "--source", dest="explicit_source", metavar="PATH")
    parser.add_argument("-t", "--target", dest="explicit_target", metavar="PATH")
    parser.add_argument("-o", "--output", dest="explicit_output", metavar="PATH")
    parser.add_argument(
        "-S",
        "--source-audio",
        type=_stream_index,
        metavar="INDEX",
        help="global stream index of the Source audio",
    )
    parser.add_argument(
        "-T",
        "--target-audio",
        type=_stream_index,
        metavar="INDEX",
        help="global stream index of the Target reference audio",
    )
    parser.add_argument(
        "--track-name",
        metavar="NAME",
        help="override the added audio track name",
    )
    parser.add_argument(
        "--language",
        type=_language_code,
        metavar="CODE",
        help="override the added audio language with an ISO 639-1 or ISO 639-2 code",
    )
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output file",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="analyze synchronization without creating an output",
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="PATH",
        help="write JSON to the exact file path (extension optional)",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="print candidates, anchors, and model diagnostics",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show detailed processing information",
    )
    output_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress progress and the final summary",
    )
    parser.add_argument(
        "--log",
        type=Path,
        metavar="PATH",
        help="append detailed diagnostics to a log file",
    )
    return parser


def build_inspect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dubgraft inspect",
        description="Inspect video and audio streams without modifying the media.",
    )
    parser.add_argument(
        "media", nargs="+", metavar="MEDIA", help="one or more media files to inspect"
    )
    return parser


def _resolve_argument(
    parser: argparse.ArgumentParser,
    role: str,
    positional: str | None,
    explicit: str | None,
    *,
    required: bool = True,
) -> Path | None:
    if positional is not None and explicit is not None:
        parser.error(f"{role} was provided more than once")
    value = explicit if explicit is not None else positional
    if value is None and required:
        parser.error(f"{role} is required")
    return Path(value) if value is not None else None


def parse_processing_config(
    argv: Sequence[str], parser: argparse.ArgumentParser | None = None
) -> ProcessingConfig:
    parser = parser or build_parser()
    arguments = parser.parse_intermixed_args(argv)
    source = _resolve_argument(
            parser, "SOURCE", arguments.positional_source, arguments.explicit_source
        )
    target = _resolve_argument(
            parser, "TARGET", arguments.positional_target, arguments.explicit_target
        )
    assert source is not None and target is not None
    return ProcessingConfig(
        source=source,
        target=target,
        output=_resolve_argument(
            parser,
            "OUTPUT",
            arguments.positional_output,
            arguments.explicit_output,
            required=not arguments.analyze_only,
        ),
        overwrite=arguments.overwrite,
        source_audio_index=arguments.source_audio,
        target_audio_index=arguments.target_audio,
        analyze_only=arguments.analyze_only,
        report=arguments.report,
        print_report=arguments.print_report,
        verbose=arguments.verbose,
        quiet=arguments.quiet,
        log=arguments.log,
        track_name=arguments.track_name,
        language=arguments.language,
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def _format_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def _format_bit_rate(bit_rate: int | None) -> str | None:
    if bit_rate is None:
        return None
    if bit_rate >= 1_000_000:
        return f"{bit_rate / 1_000_000:.2f} Mb/s"
    return f"{bit_rate / 1000:.0f} kb/s"


def _stream_flags(stream: MediaStream) -> list[str]:
    flags = []
    if stream.default:
        flags.append("default")
    if stream.forced:
        flags.append("forced")
    if stream.hearing_impaired:
        flags.append("hearing impaired")
    if stream.attached_picture:
        flags.append("attached picture")
    return flags


def _format_video(stream: MediaStream) -> str:
    codec = stream.codec
    if stream.profile:
        codec += f" {stream.profile}"
    details = [f"[{stream.index}] {codec}"]
    if stream.width is not None and stream.height is not None:
        details.append(f"{stream.width}x{stream.height}")
    if stream.frame_rate is not None:
        details.append(f"{stream.frame_rate:.3f} fps")
    if stream.pixel_format:
        details.append(stream.pixel_format)
    if stream.dolby_vision:
        dynamic_range = stream.dolby_vision
        if stream.color_transfer == "smpte2084":
            dynamic_range += " / HDR10"
        details.append(dynamic_range)
    elif stream.color_transfer == "smpte2084":
        details.append("HDR10")
    elif stream.color_transfer == "arib-std-b67":
        details.append("HLG")
    elif stream.color_space == "bt709":
        details.append("BT.709")
    details.extend(_stream_flags(stream))
    return " | ".join(details)


def _format_audio(stream: MediaStream) -> str:
    codec = stream.codec
    if stream.profile:
        profile = "Atmos" if "Atmos" in stream.profile else stream.profile
        codec += f" {profile}"
    details = [f"[{stream.index}] {codec}"]
    if stream.channel_layout:
        details.append(stream.channel_layout)
    elif stream.channels is not None:
        details.append(f"{stream.channels} channels")
    if stream.sample_rate is not None:
        details.append(f"{stream.sample_rate / 1000:g} kHz")
    bit_rate = _format_bit_rate(stream.bit_rate)
    if bit_rate:
        details.append(bit_rate)
    if stream.language:
        details.append(stream.language)
    if stream.title:
        details.append(stream.title)
    details.extend(_stream_flags(stream))
    return " | ".join(details)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    label = singular if count == 1 else plural or f"{singular}s"
    return f"{count} {label}"


def format_inspection(info: MediaInfo) -> str:
    container = info.container
    if "matroska" in container:
        container = "Matroska/WebM"
    elif "mp4" in container or container.startswith("mov"):
        container = "MP4/MOV"

    lines = [
        f"Media: {info.path.name}",
        f"Container: {container}",
        f"Duration: {_format_duration(info.duration)}",
        f"Size: {_format_size(info.size)}",
        f"Streams: {len(info.streams)}",
    ]
    bit_rate = _format_bit_rate(info.bit_rate)
    if bit_rate:
        lines.insert(4, f"Bit rate: {bit_rate}")

    videos = [stream for stream in info.streams if stream.kind == "video"]
    audios = [stream for stream in info.streams if stream.kind == "audio"]
    if videos:
        lines.extend(("", "Video"))
        lines.extend(f"  {_format_video(stream)}" for stream in videos)
    if audios:
        lines.extend(("", "Audio"))
        lines.extend(f"  {_format_audio(stream)}" for stream in audios)

    other_counts: dict[str, int] = {}
    for stream in info.streams:
        if stream.kind in {"video", "audio"}:
            continue
        other_counts[stream.kind] = other_counts.get(stream.kind, 0) + 1
    if other_counts:
        names = {
            "subtitle": ("subtitle", "subtitles"),
            "attachment": ("attachment", "attachments"),
            "data": ("data stream", "data streams"),
        }
        lines.extend(("", "Other streams"))
        for kind, count in sorted(other_counts.items()):
            singular, plural = names.get(kind, (f"{kind} stream", f"{kind} streams"))
            lines.append(f"  {_plural(count, singular, plural)} (preserved)")
    return "\n".join(lines)


def _probe_processing_media(
    parser: argparse.ArgumentParser,
    role: str,
    path: Path,
    output: RunOutput | None = None,
) -> MediaInfo:
    try:
        return probe_media(path)
    except MediaProbeError as error:
        if output is not None:
            output.error(f"could not inspect {role}: {error}")
        parser.exit(1, f"{parser.prog}: error: could not inspect {role}: {error}\n")


def _format_audio_selection_error(
    parser: argparse.ArgumentParser,
    role: str,
    option: str,
    error: AudioSelectionError,
) -> str:
    lines = [f"{parser.prog}: error: could not select {role} audio: {error}"]
    if error.candidates:
        lines.append(f"Available {role} audio streams:")
        lines.extend(f"  {_format_audio(stream)}" for stream in error.candidates)
        lines.append(f"Use {option} INDEX to select one.")
    return "\n".join(lines)


def _write_requested_report(
    config: ProcessingConfig,
    source_info: MediaInfo,
    target_info: MediaInfo,
    source_audio: MediaStream,
    target_audio: MediaStream,
    result: MatchingResult,
    *,
    status: str,
    error: str | None = None,
) -> None:
    if config.report is None:
        return
    report = build_report(
        config,
        source_info,
        target_info,
        source_audio,
        target_audio,
        result,
        status=status,
        error=error,
    )
    write_json_report(config.report, report, overwrite=config.overwrite)


def _run_inspect(argv: Sequence[str]) -> int:
    parser = build_inspect_parser()
    arguments = parser.parse_args(argv)
    inspections = []
    failed = False
    for media in arguments.media:
        path = Path(media)
        try:
            info = probe_media(path)
        except MediaProbeError as error:
            print(f"{parser.prog}: error: {path}: {error}", file=sys.stderr)
            failed = True
            continue
        inspections.append(format_inspection(info))
    if inspections:
        print("\n\n".join(inspections))
    return int(failed)


def _run_processing(
    parser: argparse.ArgumentParser,
    config: ProcessingConfig,
    output: RunOutput,
) -> int:
    try:
        with output.stage("Checking FFmpeg"):
            ffmpeg_info = validate_ffmpeg()
    except FFmpegError as error:
        output.error(str(error))
        parser.exit(1, f"{parser.prog}: error: {error}\n")
    if ffmpeg_info is not None:
        output.detail(f"FFmpeg: {ffmpeg_info.version} ({ffmpeg_info.executable})")

    with output.stage("Inspecting Source"):
        source_info = _probe_processing_media(parser, "Source", config.source, output)
    output.detail(
        f"Source: {source_info.path} | duration {_format_duration(source_info.duration)} | "
        f"{_plural(len(source_info.streams), 'stream')}"
    )
    with output.stage("Inspecting Target"):
        target_info = _probe_processing_media(parser, "Target", config.target, output)
    output.detail(
        f"Target: {target_info.path} | duration {_format_duration(target_info.duration)} | "
        f"{_plural(len(target_info.streams), 'stream')}"
    )

    selections: list[MediaStream | None] = []
    selection_errors = []
    for role, info, index, option in (
        (
            "Source",
            source_info,
            config.source_audio_index,
            "-S INDEX or --source-audio",
        ),
        (
            "Target",
            target_info,
            config.target_audio_index,
            "-T INDEX or --target-audio",
        ),
    ):
        try:
            selections.append(select_audio_stream(info, index))
        except AudioSelectionError as error:
            selections.append(None)
            selection_errors.append(
                _format_audio_selection_error(parser, role, option, error)
            )
    if selection_errors:
        output.error("; ".join(error.splitlines()[0] for error in selection_errors))
        parser.exit(2, "\n\n".join(selection_errors) + "\n")
    source_audio, target_audio = selections
    assert source_audio is not None and target_audio is not None
    output.detail(f"Source audio: {_format_audio(source_audio)}")
    output.detail(f"Target audio: {_format_audio(target_audio)}")

    try:
        with output.stage("Analyzing alignment") as stage:
            result = analyze_media(
                config,
                source_info,
                target_info,
                source_audio,
                target_audio,
                progress=lambda completed, total: stage.update(
                    f"{completed}/{total} ({completed / total:.0%})"
                ),
            )
            if result.timeline.kind is TimelineKind.INCONCLUSIVE:
                raise InconclusiveTimelineError(result.timeline, result)
    except InconclusiveTimelineError as error:
        output.error(str(error))
        if error.result is not None:
            try:
                if config.report is not None:
                    with output.stage("Writing report"):
                        _write_requested_report(
                            config,
                            source_info,
                            target_info,
                            source_audio,
                            target_audio,
                            error.result,
                            status="inconclusive",
                            error=str(error),
                        )
            except ReportError as report_error:
                output.error(str(report_error))
                if config.print_report:
                    print(
                        format_human_report(
                            config,
                            source_info,
                            target_info,
                            source_audio,
                            target_audio,
                            error.result,
                            status="inconclusive",
                            error=str(error),
                        )
                    )
                parser.exit(
                    1,
                    f"{parser.prog}: error: {format_inconclusive_error(error)}\n"
                    f"{parser.prog}: error: {report_error}\n",
                )
            if config.print_report:
                print(
                    format_human_report(
                        config,
                        source_info,
                        target_info,
                        source_audio,
                        target_audio,
                        error.result,
                        status="inconclusive",
                        error=str(error),
                    )
                )
        detail = format_inconclusive_error(error)
        parser.exit(1, f"{parser.prog}: error: {detail}\n")
    except (FFmpegError, ProcessingError) as error:
        output.error(str(error))
        parser.exit(1, f"{parser.prog}: error: {error}\n")

    output.detail(
        f"Analysis: {result.timeline.kind.value} | "
        f"{len(result.anchors)}/{result.requested_anchor_count} anchors | "
        f"coverage {result.timeline.coverage:.1%}"
    )

    if not config.analyze_only:
        try:
            with output.stage("Rendering output") as stage:
                render_media(
                    config,
                    target_info,
                    source_audio,
                    result,
                    source_duration=source_info.duration,
                    progress=lambda phase, completed, total: stage.update(
                        f"{phase} ({completed / total:.0%})"
                    ),
                )
            output.detail(f"Output: {config.output}")
        except (FFmpegError, ProcessingError) as error:
            output.error(str(error))
            try:
                if config.report is not None:
                    with output.stage("Writing report"):
                        _write_requested_report(
                            config,
                            source_info,
                            target_info,
                            source_audio,
                            target_audio,
                            result,
                            status="processing_failed",
                            error=str(error),
                        )
            except ReportError as report_error:
                output.error(str(report_error))
                if config.print_report:
                    print(
                        format_human_report(
                            config,
                            source_info,
                            target_info,
                            source_audio,
                            target_audio,
                            result,
                            status="processing_failed",
                            error=str(error),
                        )
                    )
                parser.exit(
                    1,
                    f"{parser.prog}: error: {error}\n"
                    f"{parser.prog}: error: {report_error}\n",
                )
            if config.print_report:
                print(
                    format_human_report(
                        config,
                        source_info,
                        target_info,
                        source_audio,
                        target_audio,
                        result,
                        status="processing_failed",
                        error=str(error),
                    )
                )
            parser.exit(1, f"{parser.prog}: error: {error}\n")

    status = "analysis_only" if config.analyze_only else "completed"
    try:
        if config.report is not None:
            with output.stage("Writing report"):
                _write_requested_report(
                    config,
                    source_info,
                    target_info,
                    source_audio,
                    target_audio,
                    result,
                    status=status,
                )
            output.detail(f"Report: {config.report}")
    except ReportError as error:
        output.error(str(error))
        if not config.quiet:
            if config.analyze_only:
                print(format_analysis_summary(result))
            else:
                print(
                    format_processing_summary(
                        result, config, target_info, source_audio
                    )
                )
        if config.print_report:
            if not config.quiet:
                print()
            print(
                format_human_report(
                    config,
                    source_info,
                    target_info,
                    source_audio,
                    target_audio,
                    result,
                    status=status,
                )
            )
        if config.analyze_only:
            parser.exit(1, f"{parser.prog}: error: {error}\n")
        parser.exit(
            1,
            f"{parser.prog}: error: Output was created at {config.output}, "
            f"but {error}\n",
        )

    if not config.quiet:
        if config.analyze_only:
            print(format_analysis_summary(result))
        else:
            print(format_processing_summary(result, config, target_info, source_audio))
    if config.print_report:
        if not config.quiet:
            print()
        print(
            format_human_report(
                config,
                source_info,
                target_info,
                source_audio,
                target_audio,
                result,
                status=status,
            )
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        parser.print_help()
        return 0
    if arguments[0] == "inspect":
        return _run_inspect(arguments[1:])
    config = parse_processing_config(arguments, parser)
    try:
        log_path = validate_log_path(config)
    except ConfigurationError as error:
        parser.error(str(error))

    try:
        output = RunOutput(
            verbose=config.verbose,
            quiet=config.quiet,
            log_path=log_path,
        )
    except OSError as error:
        parser.exit(1, f"{parser.prog}: error: could not open Log: {error}\n")
    with output:
        try:
            try:
                config = validate_processing_config(config)
            except ConfigurationError as error:
                output.error(str(error))
                parser.error(str(error))
            return _run_processing(parser, config, output)
        except SystemExit:
            raise
        except KeyboardInterrupt:
            output.warning("Run cancelled by user")
            parser.exit(130, f"{parser.prog}: error: interrupted by user\n")
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            output.exception(f"Unexpected failure: {detail}")
            log_hint = f"; details written to {config.log}" if config.log else ""
            parser.exit(
                1,
                f"{parser.prog}: error: unexpected failure: {detail}{log_hint}\n",
            )


if __name__ == "__main__":
    raise SystemExit(main())
