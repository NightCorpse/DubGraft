"""Typed configuration for DubGraft processing."""

import math
from dataclasses import dataclass
from pathlib import Path


ANALYSIS_SAMPLE_RATE = 22_050


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    source: Path
    target: Path
    output: Path
    overwrite: bool = False
    source_audio_index: int | None = None
    target_audio_index: int | None = None


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    fingerprint_size_seconds: float = 6.0
    scan_step_seconds: float = 20.0
    search_radius_seconds: float = 75.0
    confidence_threshold: float = 60.0


@dataclass(frozen=True, slots=True)
class TimelineConfig:
    minimum_anchor_count: int = 4
    minimum_coverage: float = 0.6
    fallback_minimum_coverage: float = 0.5
    stability_tolerance_seconds: float = 0.05
    minimum_stable_ratio: float = 0.8
    direct_tolerance_seconds: float = 0.02
    drift_tolerance_seconds: float = 0.05
    maximum_rms_residual_seconds: float = 0.05
    fallback_maximum_rms_residual_seconds: float = 0.01
    fallback_maximum_residual_seconds: float = 0.02
    fallback_duration_tolerance_seconds: float = 0.1


class ConfigurationError(ValueError):
    """Raised when DubGraft configuration violates its safety contract."""


def validate_matching_config(config: MatchingConfig) -> MatchingConfig:
    """Validate numeric matching parameters without changing their values."""
    positive_values = (
        ("fingerprint size", config.fingerprint_size_seconds),
        ("scan step", config.scan_step_seconds),
        ("search radius", config.search_radius_seconds),
    )
    for name, value in positive_values:
        if not math.isfinite(value) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive finite number")
    if (
        not math.isfinite(config.confidence_threshold)
        or config.confidence_threshold < 0
    ):
        raise ConfigurationError(
            "confidence threshold must be a non-negative finite number"
        )
    return config


def validate_timeline_config(config: TimelineConfig) -> TimelineConfig:
    """Validate temporal classification thresholds."""
    if (
        isinstance(config.minimum_anchor_count, bool)
        or not isinstance(config.minimum_anchor_count, int)
        or config.minimum_anchor_count < 2
    ):
        raise ConfigurationError("minimum anchor count must be at least 2")
    for name, value in (
        ("minimum coverage", config.minimum_coverage),
        ("fallback minimum coverage", config.fallback_minimum_coverage),
        ("minimum stable ratio", config.minimum_stable_ratio),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ConfigurationError(f"{name} must be between 0 and 1")
    for name, value in (
        ("stability tolerance", config.stability_tolerance_seconds),
        ("direct tolerance", config.direct_tolerance_seconds),
        ("drift tolerance", config.drift_tolerance_seconds),
        ("maximum RMS residual", config.maximum_rms_residual_seconds),
        (
            "fallback maximum RMS residual",
            config.fallback_maximum_rms_residual_seconds,
        ),
        ("fallback maximum residual", config.fallback_maximum_residual_seconds),
        ("fallback duration tolerance", config.fallback_duration_tolerance_seconds),
    ):
        if not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"{name} must be finite and non-negative")
    if config.fallback_minimum_coverage > config.minimum_coverage:
        raise ConfigurationError(
            "fallback minimum coverage must not exceed minimum coverage"
        )
    return config


def validate_processing_config(config: ProcessingConfig) -> ProcessingConfig:
    source = config.source.expanduser().resolve()
    target = config.target.expanduser().resolve()
    output = config.output.expanduser().resolve()

    for role, path in (("Source", source), ("Target", target)):
        if not path.exists():
            raise ConfigurationError(f"{role} file does not exist: {path}")
        if not path.is_file():
            raise ConfigurationError(f"{role} is not a file: {path}")

    if output == source:
        raise ConfigurationError("Output must not be the Source file")
    if output == target:
        raise ConfigurationError("Output must not be the Target file")
    if output.is_dir():
        raise ConfigurationError(f"Output is a directory: {output}")
    if output.exists() and not config.overwrite:
        raise ConfigurationError(
            f"Output already exists: {output}; use --overwrite to replace it"
        )
    if not output.parent.is_dir():
        raise ConfigurationError(f"Output directory does not exist: {output.parent}")

    return ProcessingConfig(
        source=source,
        target=target,
        output=output,
        overwrite=config.overwrite,
        source_audio_index=config.source_audio_index,
        target_audio_index=config.target_audio_index,
    )
