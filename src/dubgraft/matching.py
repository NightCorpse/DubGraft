"""Audio correlation, scanning, and distributed anchor selection."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from dubgraft.config import (
    ANALYSIS_SAMPLE_RATE,
    MatchingConfig,
    validate_matching_config,
)
from dubgraft.media import extract_audio_window


@dataclass(frozen=True, slots=True)
class AudioMatch:
    target_time: float
    source_time: float
    offset: float
    confidence: float


@dataclass(frozen=True, slots=True)
class MatchingResult:
    candidates: tuple[AudioMatch, ...]
    anchors: tuple[AudioMatch, ...]
    requested_anchor_count: int
    minimum_anchor_distance: float


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
) -> tuple[AudioMatch, ...]:
    """Scan the Target timeline for confident matches in the Source audio."""
    validate_matching_config(config)
    if not math.isfinite(target_duration) or target_duration <= 0:
        raise ValueError("target duration must be a positive finite number")

    candidates = []
    target_time = config.scan_step_seconds
    scan_end = target_duration - config.scan_step_seconds
    while target_time < scan_end:
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
        target_time += config.scan_step_seconds
    return tuple(candidates)


def calculate_anchor_count(duration: float) -> int:
    """Calculate the legacy-compatible desired number of anchors."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive finite number")
    return max(3, round(duration / 240.0))


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


def match_audio_streams(
    source: Path,
    source_stream_index: int,
    target: Path,
    target_stream_index: int,
    target_duration: float,
    config: MatchingConfig = MatchingConfig(),
) -> MatchingResult:
    """Run scanning and distributed anchor selection for two audio streams."""
    candidates = scan_audio_matches(
        source,
        source_stream_index,
        target,
        target_stream_index,
        target_duration,
        config,
    )
    anchor_count = calculate_anchor_count(target_duration)
    minimum_distance = calculate_minimum_anchor_distance(
        target_duration, anchor_count
    )
    anchors = select_distributed_anchors(
        candidates,
        anchor_count=anchor_count,
        minimum_distance=minimum_distance,
    )
    return MatchingResult(candidates, anchors, anchor_count, minimum_distance)
