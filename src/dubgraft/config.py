"""Typed configuration for DubGraft processing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    source: Path
    target: Path
    output: Path
    overwrite: bool = False
    source_audio_index: int | None = None
    target_audio_index: int | None = None


class ConfigurationError(ValueError):
    """Raised when processing paths violate the CLI safety contract."""


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
