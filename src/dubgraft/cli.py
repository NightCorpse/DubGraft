"""Command-line entry point for DubGraft."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dubgraft import __version__
from dubgraft.config import (
    ConfigurationError,
    ProcessingConfig,
    validate_processing_config,
)
from dubgraft.matching import MatchingResult, TimelineKind
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
    process_media,
)


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
        "-y",
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output file",
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
) -> Path:
    if positional is not None and explicit is not None:
        parser.error(f"{role} was provided more than once")
    value = explicit if explicit is not None else positional
    if value is None:
        parser.error(f"{role} is required")
    return Path(value)


def parse_processing_config(
    argv: Sequence[str], parser: argparse.ArgumentParser | None = None
) -> ProcessingConfig:
    parser = parser or build_parser()
    arguments = parser.parse_intermixed_args(argv)
    return ProcessingConfig(
        source=_resolve_argument(
            parser, "SOURCE", arguments.positional_source, arguments.explicit_source
        ),
        target=_resolve_argument(
            parser, "TARGET", arguments.positional_target, arguments.explicit_target
        ),
        output=_resolve_argument(
            parser, "OUTPUT", arguments.positional_output, arguments.explicit_output
        ),
        overwrite=arguments.overwrite,
        source_audio_index=arguments.source_audio,
        target_audio_index=arguments.target_audio,
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


def _select_processing_audio(
    parser: argparse.ArgumentParser,
    role: str,
    path: Path,
    index: int | None,
    option: str,
) -> tuple[MediaInfo, MediaStream]:
    try:
        info = probe_media(path)
    except MediaProbeError as error:
        parser.exit(1, f"{parser.prog}: error: could not inspect {role}: {error}\n")

    try:
        return info, select_audio_stream(info, index)
    except AudioSelectionError as error:
        lines = [f"{parser.prog}: error: could not select {role} audio: {error}"]
        if error.candidates:
            lines.append("Available audio streams:")
            lines.extend(f"  {_format_audio(stream)}" for stream in error.candidates)
            lines.append(f"Use {option} INDEX to select one.")
        parser.exit(2, "\n".join(lines) + "\n")


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
        config = validate_processing_config(config)
    except ConfigurationError as error:
        parser.error(str(error))
    try:
        validate_ffmpeg()
    except FFmpegError as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")
    source_info, source_audio = _select_processing_audio(
        parser,
        "Source",
        config.source,
        config.source_audio_index,
        "-S INDEX or --source-audio",
    )
    target_info, target_audio = _select_processing_audio(
        parser,
        "Target",
        config.target,
        config.target_audio_index,
        "-T INDEX or --target-audio",
    )
    try:
        result = process_media(
            config, source_info, target_info, source_audio, target_audio
        )
    except InconclusiveTimelineError as error:
        detail = format_inconclusive_error(error)
        parser.exit(1, f"{parser.prog}: error: {detail}\n")
    except (FFmpegError, ProcessingError) as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")
    print(format_processing_summary(result, config, target_info, source_audio))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
