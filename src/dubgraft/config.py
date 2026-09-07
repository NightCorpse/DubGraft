"""Typed configuration for DubGraft processing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    source: Path
    target: Path
    output: Path
    overwrite: bool = False
