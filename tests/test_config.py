import math
from pathlib import Path

import pytest

from dubgraft.config import (
    ANALYSIS_SAMPLE_RATE,
    ConfigurationError,
    MatchingConfig,
    ProcessingConfig,
    TimelineConfig,
    validate_matching_config,
    validate_processing_config,
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

    assert config.minimum_anchor_count == 4
    assert config.minimum_coverage == 0.6
    assert config.fallback_minimum_coverage == 0.5
    assert config.stability_tolerance_seconds == 0.05
    assert config.minimum_stable_ratio == 0.8
    assert config.direct_tolerance_seconds == 0.02
    assert config.drift_tolerance_seconds == 0.05
    assert config.maximum_rms_residual_seconds == 0.05
    assert config.fallback_maximum_rms_residual_seconds == 0.01
    assert config.fallback_maximum_residual_seconds == 0.02
    assert config.fallback_duration_tolerance_seconds == 0.1
    assert validate_timeline_config(config) is config


@pytest.mark.parametrize(
    "config",
    [
        TimelineConfig(minimum_anchor_count=1),
        TimelineConfig(minimum_anchor_count=2.5),  # type: ignore[arg-type]
        TimelineConfig(minimum_anchor_count=True),
        TimelineConfig(minimum_coverage=1.1),
        TimelineConfig(fallback_minimum_coverage=-0.1),
        TimelineConfig(minimum_coverage=0.5, fallback_minimum_coverage=0.6),
        TimelineConfig(minimum_stable_ratio=-0.1),
        TimelineConfig(direct_tolerance_seconds=-0.001),
        TimelineConfig(maximum_rms_residual_seconds=math.inf),
        TimelineConfig(fallback_maximum_residual_seconds=-1),
        TimelineConfig(fallback_duration_tolerance_seconds=math.inf),
    ],
)
def test_timeline_config_rejects_invalid_values(config: TimelineConfig) -> None:
    with pytest.raises(ConfigurationError):
        validate_timeline_config(config)


def test_processing_config_allows_missing_output_only_for_analysis(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    validated = validate_processing_config(
        ProcessingConfig(source, target, analyze_only=True)
    )

    assert validated.output is None


def test_processing_config_requires_output_for_rendering(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    with pytest.raises(ConfigurationError, match="Output is required"):
        validate_processing_config(ProcessingConfig(source, target))


def test_processing_config_resolves_log_path(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    validated = validate_processing_config(
        ProcessingConfig(source, target, analyze_only=True, log=Path("run.log"))
    )

    assert validated.log == Path("run.log").resolve()


def test_processing_config_rejects_log_collisions(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    source.touch()
    target.touch()

    with pytest.raises(ConfigurationError, match="Log must not replace"):
        validate_processing_config(
            ProcessingConfig(source, target, analyze_only=True, log=source)
        )
