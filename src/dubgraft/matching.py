"""Audio correlation, scanning, and distributed anchor selection."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from dubgraft.config import (
    ANALYSIS_SAMPLE_RATE,
    MatchingConfig,
    TimelineConfig,
    validate_matching_config,
    validate_timeline_config,
)
from dubgraft.media import extract_audio_window


@dataclass(frozen=True, slots=True)
class AudioMatch:
    target_time: float
    source_time: float
    offset: float
    confidence: float


class TimelineKind(str, Enum):
    DIRECT = "direct"
    STATIC = "static"
    DRIFT = "drift"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class TimelineAnalysis:
    kind: TimelineKind
    median_offset: float | None
    slope: float | None
    intercept: float | None
    rms_residual: float | None
    maximum_residual: float | None
    coverage: float
    stable_ratio: float
    drift_over_duration: float | None
    reason: str | None = None
    used_duration_fallback: bool = False
    source_duration_error: float | None = None


@dataclass(frozen=True, slots=True)
class MatchingResult:
    candidates: tuple[AudioMatch, ...]
    anchors: tuple[AudioMatch, ...]
    requested_anchor_count: int
    minimum_anchor_distance: float
    timeline: TimelineAnalysis


def correlate_audio(
    fingerprint: NDArray[np.float32],
    search_window: NDArray[np.float32],
    *,
    target_time: float,
    search_start: float,
) -> AudioMatch | None:
    """Locate a target fingerprint in a Source search window."""
    if fingerprint.ndim != 1 or search_window.ndim != 1:
        raise ValueError("audio correlation requires one-dimensional samples")
    if fingerprint.size == 0 or search_window.size <= fingerprint.size:
        return None

    correlation = signal.correlate(search_window, fingerprint, mode="valid")
    absolute = np.abs(correlation)
    mean = float(np.mean(absolute))
    if mean == 0 or not math.isfinite(mean):
        return None

    peak_index = int(np.argmax(absolute))
    confidence = float(absolute[peak_index] / mean)
    if not math.isfinite(confidence):
        return None
    source_time = search_start + peak_index / ANALYSIS_SAMPLE_RATE
    return AudioMatch(
        target_time=target_time,
        source_time=source_time,
        offset=source_time - target_time,
        confidence=confidence,
    )


def scan_audio_matches(
    source: Path,
    source_stream_index: int,
    target: Path,
    target_stream_index: int,
    target_duration: float,
    config: MatchingConfig = MatchingConfig(),
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[AudioMatch, ...]:
    """Scan the Target timeline for confident matches in the Source audio."""
    validate_matching_config(config)
    if not math.isfinite(target_duration) or target_duration <= 0:
        raise ValueError("target duration must be a positive finite number")

    candidates = []
    scan_times = []
    target_time = config.scan_step_seconds
    scan_end = target_duration - config.scan_step_seconds
    while target_time < scan_end:
        scan_times.append(target_time)
        target_time += config.scan_step_seconds
    for completed, target_time in enumerate(scan_times, start=1):
        fingerprint = extract_audio_window(
            target,
            target_stream_index,
            target_time,
            config.fingerprint_size_seconds,
        )
        search_start = max(0.0, target_time - config.search_radius_seconds)
        search_window = extract_audio_window(
            source,
            source_stream_index,
            search_start,
            config.fingerprint_size_seconds + 2 * config.search_radius_seconds,
        )
        match = correlate_audio(
            fingerprint,
            search_window,
            target_time=target_time,
            search_start=search_start,
        )
        if match is not None and match.confidence >= config.confidence_threshold:
            candidates.append(match)
        if progress is not None:
            progress(completed, len(scan_times))
    return tuple(candidates)


def calculate_anchor_count(duration: float) -> int:
    """Calculate the desired number of distributed anchors."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive finite number")
    return max(4, round(duration / 240.0))


def calculate_minimum_anchor_distance(duration: float, anchor_count: int) -> float:
    """Calculate how far apart selected anchors must be on the Target timeline."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive finite number")
    if anchor_count <= 0:
        raise ValueError("anchor count must be positive")
    return float(max(20, int(0.6 * duration / anchor_count)))


def select_distributed_anchors(
    candidates: tuple[AudioMatch, ...],
    *,
    anchor_count: int,
    minimum_distance: float,
) -> tuple[AudioMatch, ...]:
    """Select the strongest candidates while preventing temporal clustering."""
    if anchor_count <= 0:
        raise ValueError("anchor count must be positive")
    if not math.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("minimum anchor distance must be finite and non-negative")

    selected = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if all(
            abs(candidate.target_time - anchor.target_time) > minimum_distance
            for anchor in selected
        ):
            selected.append(candidate)
            if len(selected) == anchor_count:
                break
    return tuple(sorted(selected, key=lambda item: item.target_time))


def analyze_timeline(
    anchors: tuple[AudioMatch, ...],
    target_duration: float,
    config: TimelineConfig = TimelineConfig(),
    *,
    source_duration: float | None = None,
) -> TimelineAnalysis:
    """Fit and classify the temporal relationship represented by audio anchors."""
    validate_timeline_config(config)
    if not math.isfinite(target_duration) or target_duration <= 0:
        raise ValueError("target duration must be a positive finite number")
    if source_duration is not None and (
        not math.isfinite(source_duration) or source_duration <= 0
    ):
        raise ValueError("source duration must be a positive finite number")
    for anchor in anchors:
        values = (
            anchor.target_time,
            anchor.source_time,
            anchor.offset,
            anchor.confidence,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("anchors must contain only finite values")
        if not 0 <= anchor.target_time <= target_duration:
            raise ValueError("anchor target time must be within the Target duration")
        if not math.isclose(
            anchor.offset,
            anchor.source_time - anchor.target_time,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("anchor offset must equal Source time minus Target time")

    if not anchors:
        return TimelineAnalysis(
            TimelineKind.INCONCLUSIVE,
            None,
            None,
            None,
            None,
            None,
            0.0,
            0.0,
            None,
            "no anchors were found",
        )

    target_times = np.array([anchor.target_time for anchor in anchors])
    source_times = np.array([anchor.source_time for anchor in anchors])
    offsets = source_times - target_times
    median_offset = float(np.median(offsets))
    stable_ratio = float(
        np.mean(np.abs(offsets - median_offset) <= config.stability_tolerance_seconds)
    )
    coverage = float((np.max(target_times) - np.min(target_times)) / target_duration)

    if len(anchors) < config.minimum_anchor_count:
        return TimelineAnalysis(
            TimelineKind.INCONCLUSIVE,
            median_offset,
            None,
            None,
            None,
            None,
            coverage,
            stable_ratio,
            None,
            f"at least {config.minimum_anchor_count} anchors are required",
        )
    if np.unique(target_times).size < 2:
        raise ValueError("anchors must contain at least two distinct Target times")

    stable_mask = (
        np.abs(offsets - median_offset) <= config.stability_tolerance_seconds
    )
    stable_target_times = target_times[stable_mask]
    use_stable_consensus = (
        stable_ratio >= config.minimum_stable_ratio
        and stable_target_times.size >= config.minimum_anchor_count
        and np.unique(stable_target_times).size >= 2
    )
    if use_stable_consensus:
        fit_target_times = target_times[stable_mask]
        fit_source_times = source_times[stable_mask]
    else:
        fit_target_times = target_times
        fit_source_times = source_times

    coverage = float(
        (np.max(fit_target_times) - np.min(fit_target_times)) / target_duration
    )

    slope, intercept = np.polyfit(fit_target_times, fit_source_times, 1)
    predicted = slope * fit_target_times + intercept
    residuals = fit_source_times - predicted
    rms_residual = float(np.sqrt(np.mean(np.square(residuals))))
    maximum_residual = float(np.max(np.abs(residuals)))
    drift_over_duration = float((slope - 1.0) * target_duration)

    values = (slope, intercept, rms_residual, maximum_residual, drift_over_duration)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("timeline regression produced non-finite values")

    predicted_source_end = float(slope * target_duration + intercept)
    duration_error = (
        abs(predicted_source_end - source_duration)
        if source_duration is not None
        else None
    )
    used_duration_fallback = (
        coverage < config.minimum_coverage
        and coverage >= config.fallback_minimum_coverage
        and rms_residual <= config.fallback_maximum_rms_residual_seconds
        and maximum_residual <= config.fallback_maximum_residual_seconds
        and duration_error is not None
        and duration_error <= config.fallback_duration_tolerance_seconds
    )

    reason = None
    if coverage < config.minimum_coverage and not used_duration_fallback:
        kind = TimelineKind.INCONCLUSIVE
        reason = "anchors do not cover enough of the Target timeline"
    elif rms_residual > config.maximum_rms_residual_seconds:
        kind = TimelineKind.INCONCLUSIVE
        reason = "anchor residuals are too large for a linear timeline"
    elif abs(drift_over_duration) > config.drift_tolerance_seconds:
        kind = TimelineKind.DRIFT
    elif stable_ratio < config.minimum_stable_ratio:
        kind = TimelineKind.INCONCLUSIVE
        reason = "anchor offsets are not sufficiently stable"
    elif abs(median_offset) <= config.direct_tolerance_seconds:
        kind = TimelineKind.DIRECT
    else:
        kind = TimelineKind.STATIC

    return TimelineAnalysis(
        kind,
        median_offset,
        float(slope),
        float(intercept),
        rms_residual,
        maximum_residual,
        coverage,
        stable_ratio,
        drift_over_duration,
        reason,
        used_duration_fallback,
        duration_error,
    )


def match_audio_streams(
    source: Path,
    source_stream_index: int,
    target: Path,
    target_stream_index: int,
    target_duration: float,
    config: MatchingConfig = MatchingConfig(),
    timeline_config: TimelineConfig = TimelineConfig(),
    *,
    source_duration: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MatchingResult:
    """Run scanning and distributed anchor selection for two audio streams."""
    validate_timeline_config(timeline_config)
    candidates = scan_audio_matches(
        source,
        source_stream_index,
        target,
        target_stream_index,
        target_duration,
        config,
        progress=progress,
    )
    anchor_count = max(
        calculate_anchor_count(target_duration), timeline_config.minimum_anchor_count
    )
    minimum_distance = calculate_minimum_anchor_distance(
        target_duration, anchor_count
    )
    anchors = select_distributed_anchors(
        candidates,
        anchor_count=anchor_count,
        minimum_distance=minimum_distance,
    )
    timeline = analyze_timeline(
        anchors,
        target_duration,
        timeline_config,
        source_duration=source_duration,
    )
    return MatchingResult(candidates, anchors, anchor_count, minimum_distance, timeline)
