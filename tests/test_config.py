import math

import pytest

from dubgraft.config import (
    ANALYSIS_SAMPLE_RATE,
    ConfigurationError,
    MatchingConfig,
    validate_matching_config,
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
