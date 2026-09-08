from pathlib import Path

import numpy as np
import pytest

from dubgraft.config import ANALYSIS_SAMPLE_RATE, MatchingConfig, TimelineConfig
from dubgraft.matching import (
    AudioMatch,
    TimelineKind,
    analyze_timeline,
    calculate_anchor_count,
    calculate_minimum_anchor_distance,
    correlate_audio,
    match_audio_streams,
    scan_audio_matches,
    select_distributed_anchors,
)


def test_correlate_audio_locates_fingerprint_in_search_window() -> None:
    generator = np.random.default_rng(42)
    fingerprint = generator.normal(size=2_000).astype(np.float32)
    search_window = generator.normal(scale=0.01, size=10_000).astype(np.float32)
    peak_index = 3_500
    search_window[peak_index : peak_index + fingerprint.size] += fingerprint

    match = correlate_audio(
        fingerprint,
        search_window,
        target_time=100,
        search_start=90,
    )

    assert match is not None
    assert match.source_time == pytest.approx(90 + peak_index / ANALYSIS_SAMPLE_RATE)
    assert match.offset == pytest.approx(match.source_time - 100)
    assert match.confidence > 1


def test_correlate_audio_ignores_flat_or_incomplete_samples() -> None:
    fingerprint = np.zeros(100, dtype=np.float32)

    assert (
        correlate_audio(
            fingerprint,
            np.zeros(200, dtype=np.float32),
            target_time=20,
            search_start=0,
        )
        is None
    )
    assert (
        correlate_audio(
            fingerprint,
            np.zeros(100, dtype=np.float32),
            target_time=20,
            search_start=0,
        )
        is None
    )


def test_scan_audio_matches_filters_candidates_by_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = np.zeros(100, dtype=np.float32)
    fingerprint[50] = 1
    strong_window = np.zeros(500, dtype=np.float32)
    strong_window[200:300] = fingerprint
    calls: list[tuple[Path, int, float, float]] = []

    def extract(path: Path, index: int, start: float, duration: float) -> np.ndarray:
        calls.append((path, index, start, duration))
        return fingerprint if path == Path("target.mkv") else strong_window

    monkeypatch.setattr("dubgraft.matching.extract_audio_window", extract)
    config = MatchingConfig(
        fingerprint_size_seconds=6,
        scan_step_seconds=20,
        search_radius_seconds=5,
        confidence_threshold=8,
    )

    matches = scan_audio_matches(
        Path("source.mkv"), 1, Path("target.mkv"), 2, 70, config
    )

    assert len(matches) == 2
    assert [match.target_time for match in matches] == [20, 40]
    assert calls[0] == (Path("target.mkv"), 2, 20, 6)
    assert calls[1] == (Path("source.mkv"), 1, 15, 16)


def test_scan_audio_matches_skips_an_unavailable_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = np.zeros(100, dtype=np.float32)
    fingerprint[50] = 1
    strong_window = np.zeros(500, dtype=np.float32)
    strong_window[200:300] = fingerprint
    source_windows = iter((np.array([], dtype=np.float32), strong_window))

    def extract(path: Path, index: int, start: float, duration: float) -> np.ndarray:
        return fingerprint if path == Path("target.mkv") else next(source_windows)

    monkeypatch.setattr("dubgraft.matching.extract_audio_window", extract)
    config = MatchingConfig(
        scan_step_seconds=20,
        search_radius_seconds=5,
        confidence_threshold=8,
    )

    matches = scan_audio_matches(
        Path("source.mkv"), 1, Path("target.mkv"), 2, 70, config
    )

    assert [match.target_time for match in matches] == [40]


def test_anchor_formulas_match_legacy_behavior() -> None:
    anchor_count = calculate_anchor_count(3218)

    assert anchor_count == 13
    assert calculate_minimum_anchor_distance(3218, anchor_count) == 148
    assert calculate_anchor_count(600) == 4


def test_select_distributed_anchors_keeps_strong_separated_candidates() -> None:
    candidates = (
        AudioMatch(260, 260.012, 0.012, 100.8),
        AudioMatch(280, 280.012, 0.012, 86.5),
        AudioMatch(520, 520.012, 0.012, 131.1),
    )

    anchors = select_distributed_anchors(
        candidates, anchor_count=3, minimum_distance=148
    )

    assert [anchor.target_time for anchor in anchors] == [260, 520]


def timeline_anchors(*, slope: float = 1.0, intercept: float = 0.0) -> tuple[AudioMatch, ...]:
    return tuple(
        AudioMatch(
            target_time=target_time,
            source_time=slope * target_time + intercept,
            offset=(slope - 1) * target_time + intercept,
            confidence=100,
        )
        for target_time in (100.0, 400.0, 700.0, 900.0)
    )


def test_analyze_timeline_classifies_direct_sync() -> None:
    result = analyze_timeline(timeline_anchors(intercept=0.007), 1000)

    assert result.kind is TimelineKind.DIRECT
    assert result.median_offset == pytest.approx(0.007)
    assert result.slope == pytest.approx(1)
    assert result.coverage == pytest.approx(0.8)
    assert result.stable_ratio == pytest.approx(1)
    assert result.rms_residual == pytest.approx(0, abs=1e-12)


def test_analyze_timeline_classifies_static_offset() -> None:
    result = analyze_timeline(timeline_anchors(intercept=2), 1000)

    assert result.kind is TimelineKind.STATIC
    assert result.median_offset == pytest.approx(2)
    assert result.drift_over_duration == pytest.approx(0, abs=1e-9)


def test_analyze_timeline_classifies_linear_drift() -> None:
    result = analyze_timeline(timeline_anchors(slope=1.001), 1000)

    assert result.kind is TimelineKind.DRIFT
    assert result.slope == pytest.approx(1.001)
    assert result.drift_over_duration == pytest.approx(1)
    assert result.rms_residual == pytest.approx(0, abs=1e-9)


def test_analyze_timeline_is_inconclusive_with_too_few_anchors() -> None:
    result = analyze_timeline(timeline_anchors()[:3], 1000)

    assert result.kind is TimelineKind.INCONCLUSIVE
    assert result.reason == "at least 4 anchors are required"
    assert result.slope is None


def test_analyze_timeline_is_inconclusive_without_enough_coverage() -> None:
    anchors = tuple(
        AudioMatch(time, time, 0, 100) for time in (100.0, 150.0, 200.0, 300.0)
    )

    result = analyze_timeline(anchors, 1000)

    assert result.kind is TimelineKind.INCONCLUSIVE
    assert result.reason == "anchors do not cover enough of the Target timeline"


def test_analyze_timeline_rejects_non_linear_anchor_residuals() -> None:
    anchors = (
        AudioMatch(100, 100, 0, 100),
        AudioMatch(400, 402, 2, 100),
        AudioMatch(700, 699, -1, 100),
        AudioMatch(900, 903, 3, 100),
    )

    result = analyze_timeline(anchors, 1000)

    assert result.kind is TimelineKind.INCONCLUSIVE
    assert result.reason == "anchor residuals are too large for a linear timeline"


def test_analyze_timeline_uses_strict_duration_fallback() -> None:
    slope = 0.999
    intercept = 0.003
    anchors = tuple(
        AudioMatch(
            target_time=time,
            source_time=slope * time + intercept,
            offset=(slope - 1) * time + intercept,
            confidence=100,
        )
        for time in (200.0, 560.0, 1000.0, 1760.0)
    )
    source_duration = slope * 2863.82 + intercept

    result = analyze_timeline(
        anchors,
        2863.82,
        source_duration=source_duration,
    )

    assert result.kind is TimelineKind.DRIFT
    assert result.coverage == pytest.approx(0.544727, abs=1e-6)
    assert result.used_duration_fallback is True


@pytest.mark.parametrize("duration_delta", [-0.101, 0.101])
def test_analyze_timeline_rejects_fallback_with_incompatible_duration(
    duration_delta: float,
) -> None:
    slope = 0.999
    anchors = tuple(
        AudioMatch(time, slope * time, (slope - 1) * time, 100)
        for time in (200.0, 560.0, 1000.0, 1760.0)
    )

    result = analyze_timeline(
        anchors,
        2863.82,
        source_duration=slope * 2863.82 + duration_delta,
    )

    assert result.kind is TimelineKind.INCONCLUSIVE
    assert result.used_duration_fallback is False


def test_analyze_timeline_ignores_a_minority_offset_outlier() -> None:
    anchors = tuple(
        AudioMatch(time, time, 0, 100) for time in range(100, 1000, 100)
    ) + (AudioMatch(1000, 1000.15, 0.15, 100),)

    result = analyze_timeline(anchors, 1100)

    assert result.kind is TimelineKind.DIRECT
    assert result.stable_ratio == pytest.approx(0.9)
    assert result.slope == pytest.approx(1)


def test_analyze_timeline_excludes_outliers_from_coverage() -> None:
    anchors = tuple(
        AudioMatch(time, time, 0, 100) for time in range(100, 301, 25)
    ) + (AudioMatch(900, 900.15, 0.15, 100),)

    result = analyze_timeline(anchors, 1000)

    assert result.kind is TimelineKind.INCONCLUSIVE
    assert result.coverage == pytest.approx(0.2)
    assert result.reason == "anchors do not cover enough of the Target timeline"


def test_analyze_timeline_falls_back_when_stable_consensus_is_too_small() -> None:
    anchors = (
        AudioMatch(100, 100.1, 0.1, 100),
        AudioMatch(350, 350.15, 0.15, 100),
        AudioMatch(600, 600.2, 0.2, 100),
        AudioMatch(900, 900.3, 0.3, 100),
    )

    result = analyze_timeline(
        anchors,
        1000,
        TimelineConfig(minimum_stable_ratio=0, stability_tolerance_seconds=0),
    )

    assert result.slope is not None
    assert result.coverage == pytest.approx(0.8)


def test_analyze_timeline_rejects_inconsistent_anchor_offset() -> None:
    for anchor in (
        AudioMatch(100, 102, 0, 100),
        AudioMatch(100, 1_000_000_100, 999_999_999.5, 100),
    ):
        with pytest.raises(ValueError, match="offset must equal"):
            analyze_timeline((anchor,), 1000)


def test_match_audio_streams_applies_timeline_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = timeline_anchors(intercept=0.007)
    monkeypatch.setattr(
        "dubgraft.matching.scan_audio_matches", lambda *args, **kwargs: anchors
    )

    result = match_audio_streams(
        Path("source.mkv"),
        1,
        Path("target.mkv"),
        2,
        1000,
        timeline_config=TimelineConfig(direct_tolerance_seconds=0.005),
    )

    assert result.timeline.kind is TimelineKind.STATIC


def test_match_audio_streams_requests_configured_minimum_anchor_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = tuple(
        AudioMatch(time, time, 0, 100) for time in (100, 300, 500, 700, 900)
    )
    monkeypatch.setattr(
        "dubgraft.matching.scan_audio_matches", lambda *args, **kwargs: anchors
    )

    result = match_audio_streams(
        Path("source.mkv"),
        1,
        Path("target.mkv"),
        2,
        1000,
        timeline_config=TimelineConfig(minimum_anchor_count=5),
    )

    assert result.requested_anchor_count == 5
    assert len(result.anchors) == 5
    assert result.timeline.kind is TimelineKind.DIRECT
