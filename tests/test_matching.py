from pathlib import Path

import numpy as np
import pytest

from dubgraft.config import ANALYSIS_SAMPLE_RATE, MatchingConfig
from dubgraft.matching import (
    AudioMatch,
    calculate_anchor_count,
    calculate_minimum_anchor_distance,
    correlate_audio,
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
