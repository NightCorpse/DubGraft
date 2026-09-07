import math

import pytest

from dubgraft.config import (
    ANALYSIS_SAMPLE_RATE,
    ConfigurationError,
    MatchingConfig,
    TimelineConfig,
    validate_matching_config,
    validate_timeline_config,
)


def test_matching_config_uses_validated_legacy_defaults() -> None:
    config = MatchingConfig()

    assert config.fingerprint_size_seconds == 6
    assert config.scan_step_seconds == 20
    assert config.search_radius_seconds == 75
    assert config.confidence_threshold == 60
    assert ANALYSIS_SAMPLE_RATE == 22_050
    assert validate_matching_config(config) is config


@pytest.mark.parametrize(
    "config",
    [
        MatchingConfig(fingerprint_size_seconds=0),
        MatchingConfig(scan_step_seconds=-1),
        MatchingConfig(search_radius_seconds=math.inf),
        MatchingConfig(confidence_threshold=-1),
    ],
)
def test_matching_config_rejects_invalid_values(config: MatchingConfig) -> None:
    with pytest.raises(ConfigurationError):
        validate_matching_config(config)


@pytest.mark.parametrize("threshold", [0, 1, 7, 8, 20, 60])
def test_matching_config_accepts_safe_confidence_thresholds(
    threshold: float,
) -> None:
    config = MatchingConfig(confidence_threshold=threshold)

    assert validate_matching_config(config) is config


def test_timeline_config_uses_validated_defaults() -> None:
    config = TimelineConfig()

    assert config.minimum_anchor_count == 3
    assert config.minimum_coverage == 0.6
    assert config.stability_tolerance_seconds == 0.05
    assert config.minimum_stable_ratio == 0.8
    assert config.direct_tolerance_seconds == 0.02
    assert config.drift_tolerance_seconds == 0.05
    assert config.maximum_rms_residual_seconds == 0.05
    assert validate_timeline_config(config) is config


@pytest.mark.parametrize(
    "config",
    [
        TimelineConfig(minimum_anchor_count=1),
        TimelineConfig(minimum_anchor_count=2.5),  # type: ignore[arg-type]
        TimelineConfig(minimum_anchor_count=True),
        TimelineConfig(minimum_coverage=1.1),
        TimelineConfig(minimum_stable_ratio=-0.1),
        TimelineConfig(direct_tolerance_seconds=-0.001),
        TimelineConfig(maximum_rms_residual_seconds=math.inf),
    ],
)
def test_timeline_config_rejects_invalid_values(config: TimelineConfig) -> None:
    with pytest.raises(ConfigurationError):
        validate_timeline_config(config)
