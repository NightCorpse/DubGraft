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
