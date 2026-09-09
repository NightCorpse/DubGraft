"""Structured and human-readable matching reports."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dubgraft import __version__
from dubgraft.config import (
    ANALYSIS_SAMPLE_RATE,
    MatchingConfig,
    ProcessingConfig,
    TimelineConfig,
)
from dubgraft.matching import AudioMatch, MatchingResult, TimelineKind
from dubgraft.media import MediaInfo, MediaStream


class ReportError(RuntimeError):
    """Raised when a requested report cannot be published safely."""


def _added_audio(output_info: MediaInfo | None) -> MediaStream | None:
    if output_info is None:
        return None
    if output_info.added_audio_stream_index is not None:
        return next(
            (
                stream
                for stream in output_info.streams
                if stream.index == output_info.added_audio_stream_index
            ),
            None,
        )
    return next(
        (stream for stream in reversed(output_info.streams) if stream.kind == "audio"),
        None,
    )


def _match_data(match: AudioMatch) -> dict[str, float]:
    return {
        "target_time_seconds": match.target_time,
        "source_time_seconds": match.source_time,
        "offset_seconds": match.offset,
        "confidence": match.confidence,
    }


def build_report(
    config: ProcessingConfig,
    source_info: MediaInfo,
    target_info: MediaInfo,
    source_audio: MediaStream,
    target_audio: MediaStream,
    result: MatchingResult,
    *,
    status: str,
    error: str | None = None,
    output_info: MediaInfo | None = None,
    matching_config: MatchingConfig | None = None,
    timeline_config: TimelineConfig | None = None,
) -> dict[str, Any]:
    """Build the stable JSON-compatible representation of one analysis."""
    matching_config = matching_config or config.matching_config
    timeline_config = timeline_config or config.timeline_config
    timeline = result.timeline
    strategy = None
    if timeline.kind in {TimelineKind.DIRECT, TimelineKind.STATIC}:
        strategy = "stream_copy"
    elif timeline.kind is TimelineKind.DRIFT:
        strategy = "eac3_640k"

    added_audio = _added_audio(output_info)
    return {
        "schema_version": 1,
        "dubgraft_version": __version__,
        "status": status,
        "error": error,
        "mode": "analyze_only" if config.analyze_only else "process",
        "media": {
            "source": {
                "path": str(source_info.path),
                "container": source_info.container,
                "duration_seconds": source_info.duration,
                "selected_audio": asdict(source_audio),
            },
            "target": {
                "path": str(target_info.path),
                "container": target_info.container,
                "duration_seconds": target_info.duration,
                "stream_count": len(target_info.streams),
                "selected_audio": asdict(target_audio),
            },
            "output": str(config.output) if config.output is not None else None,
        },
        "parameters": {
            "analysis_sample_rate_hz": ANALYSIS_SAMPLE_RATE,
            "matching": asdict(matching_config),
            "timeline": asdict(timeline_config),
            "requested_anchor_count": result.requested_anchor_count,
            "minimum_anchor_distance_seconds": result.minimum_anchor_distance,
        },
        "formulas": {
            "timeline": "source_time = slope * target_time + intercept",
            "offset": "source_time - target_time",
            "confidence": "absolute_correlation_peak / mean_absolute_correlation",
            "coverage": "(last_anchor_target_time - first_anchor_target_time) / target_duration",
        },
        "matching": {
            "candidate_count": len(result.candidates),
            "anchor_count": len(result.anchors),
            "candidates": [_match_data(match) for match in result.candidates],
            "anchors": [_match_data(match) for match in result.anchors],
        },
        "timeline": {
            "classification": timeline.kind.value,
            "reason": timeline.reason,
            "median_offset_seconds": timeline.median_offset,
            "slope": timeline.slope,
            "intercept_seconds": timeline.intercept,
            "rms_residual_seconds": timeline.rms_residual,
            "maximum_residual_seconds": timeline.maximum_residual,
            "coverage": timeline.coverage,
            "stable_ratio": timeline.stable_ratio,
            "drift_over_duration_seconds": timeline.drift_over_duration,
            "used_duration_fallback": timeline.used_duration_fallback,
            "source_duration_error_seconds": timeline.source_duration_error,
        },
        "processing": {
            "strategy": strategy,
            "rendered": status == "completed",
            "output_validated": added_audio is not None,
            "target_streams_preserved": (
                len(target_info.streams) if output_info is not None else None
            ),
            "source_audio_codec": source_audio.codec,
            "source_audio_channels": source_audio.channels,
            "source_audio_channel_layout": source_audio.channel_layout,
            "source_audio_sample_rate_hz": source_audio.sample_rate,
            "requested_metadata_overrides": {
                "language": config.language,
                "title": config.track_name,
            },
            "added_audio": asdict(added_audio) if added_audio is not None else None,
        },
    }


def write_json_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    """Publish a JSON report without partially replacing an existing file."""
    try:
        with tempfile.TemporaryDirectory(prefix=".dubgraft-report-", dir=path.parent) as temp:
            temporary_report = Path(temp) / path.name
            temporary_report.write_text(
                json.dumps(report, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            if overwrite:
                temporary_report.replace(path)
            else:
                try:
                    os.link(temporary_report, path)
                except FileExistsError as error:
                    raise ReportError(
                        f"Report already exists: {path}; use --overwrite to replace it"
                    ) from error
    except ReportError:
        raise
    except OSError as error:
        raise ReportError(f"could not write report {path}: {error}") from error


def _value(value: float | None, digits: int = 6) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def format_human_report(
    config: ProcessingConfig,
    source_info: MediaInfo,
    target_info: MediaInfo,
    source_audio: MediaStream,
    target_audio: MediaStream,
    result: MatchingResult,
    *,
    status: str,
    error: str | None = None,
    output_info: MediaInfo | None = None,
    matching_config: MatchingConfig | None = None,
    timeline_config: TimelineConfig | None = None,
) -> str:
    """Format candidates, anchors, model quality, and processing decisions."""
    matching_config = matching_config or config.matching_config
    timeline_config = timeline_config or config.timeline_config
    timeline = result.timeline
    added_audio = _added_audio(output_info)
    output = str(config.output) if config.output is not None else "not requested"
    lines = [
        "DubGraft Analysis Report",
        f"Status: {status}",
        f"Mode: {'analyze only' if config.analyze_only else 'process'}",
        f"Source: {source_info.path} | audio stream {source_audio.index}",
        f"Target: {target_info.path} | audio stream {target_audio.index}",
        f"Output: {output}",
    ]
    if error:
        lines.append(f"Error: {error}")

    source_format = _audio_format(source_audio)
    target_format = _audio_format(target_audio)
    lines.extend(
        [
            "",
            "Selected audio",
            f"  Source: {source_format}",
            f"  Target reference: {target_format}",
            "",
            "Parameters",
            f"  Analysis sample rate: {ANALYSIS_SAMPLE_RATE} Hz",
            f"  Fingerprint size: {matching_config.fingerprint_size_seconds:.3f}s",
            f"  Scan step: {matching_config.scan_step_seconds:.3f}s",
            f"  Search radius: {matching_config.search_radius_seconds:.3f}s",
            f"  Minimum confidence: {matching_config.confidence_threshold:.3f}",
            f"  Anchors requested: {result.requested_anchor_count}",
            f"  Anchor gap: {result.minimum_anchor_distance:.3f}s",
            f"  Minimum coverage: {timeline_config.minimum_coverage:.3%}",
            f"  Fallback minimum coverage: {timeline_config.fallback_minimum_coverage:.3%}",
            f"  Direct limit: {timeline_config.direct_tolerance_seconds * 1000:.3f}ms",
            f"  Drift tolerance: {timeline_config.drift_tolerance_seconds:.3f}s",
            f"  Maximum RMS residual: {timeline_config.maximum_rms_residual_seconds:.3f}s",
            "",
            "Formulas",
            "  Timeline: source_time = slope * target_time + intercept",
            "  Offset: source_time - target_time",
            "  Confidence: absolute correlation peak / mean absolute correlation",
            "  Coverage: anchor span on Target / Target duration",
            "",
            "Timeline",
            f"  Classification: {timeline.kind.value}",
            f"  Median offset: {_value(timeline.median_offset)}s",
            f"  Slope: {_value(timeline.slope, 12)}",
            f"  Intercept: {_value(timeline.intercept)}s",
            f"  Total drift: {_value(timeline.drift_over_duration)}s",
            f"  Coverage: {timeline.coverage:.3%}",
            f"  Stable ratio: {timeline.stable_ratio:.3%}",
            f"  RMS residual: {_value(timeline.rms_residual)}s",
            f"  Maximum residual: {_value(timeline.maximum_residual)}s",
            f"  Duration fallback: {'yes' if timeline.used_duration_fallback else 'no'}",
            f"  Source duration error: {_value(timeline.source_duration_error)}s",
        ]
    )
    if timeline.reason:
        lines.append(f"  Reason: {timeline.reason}")

    strategy = "none"
    if timeline.kind in {TimelineKind.DIRECT, TimelineKind.STATIC}:
        strategy = "stream copy"
    elif timeline.kind is TimelineKind.DRIFT:
        strategy = "continuous retiming and E-AC-3 640 kb/s"
    lines.extend(
        [
            "",
            "Processing",
            f"  Strategy: {strategy}",
            f"  Rendered: {'yes' if status == 'completed' else 'no'}",
            f"  Output validated: {'yes' if added_audio is not None else 'no'}",
            "  Target streams preserved: "
            + (
                str(len(target_info.streams))
                if output_info is not None
                else "not validated"
            ),
        ]
    )
    overrides = []
    if config.language is not None:
        overrides.append(f"language={config.language}")
    if config.track_name is not None:
        overrides.append(f"title={config.track_name}")
    if overrides:
        lines.append(f"  Requested metadata overrides: {' | '.join(overrides)}")
    if added_audio is not None:
        lines.append(f"  Added audio: {_audio_format(added_audio)}")

    lines.extend(("", f"Candidates ({len(result.candidates)})"))
    lines.extend(_format_matches(result.candidates))
    lines.extend(("", f"Selected anchors ({len(result.anchors)})"))
    lines.extend(_format_matches(result.anchors))
    return "\n".join(lines)


def _audio_format(stream: MediaStream) -> str:
    details = [f"stream {stream.index}", stream.codec]
    if stream.profile:
        details.append(stream.profile)
    if stream.channel_layout:
        details.append(stream.channel_layout)
    elif stream.channels is not None:
        details.append(f"{stream.channels} channels")
    if stream.sample_rate is not None:
        details.append(f"{stream.sample_rate} Hz")
    if stream.language:
        details.append(stream.language)
    if stream.title:
        details.append(stream.title)
    return " | ".join(details)


def _format_matches(matches: tuple[AudioMatch, ...]) -> list[str]:
    if not matches:
        return ["  none"]
    lines = ["  #  Target (s)   Source (s)   Offset (s)   Confidence"]
    for index, match in enumerate(matches, 1):
        lines.append(
            f"  {index:>2}  {match.target_time:>10.3f}   "
            f"{match.source_time:>10.3f}   {match.offset:>+10.3f}   "
            f"{match.confidence:>10.3f}"
        )
    return lines
