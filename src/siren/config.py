from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Device = Literal["auto", "cpu", "cuda"]


@dataclass(slots=True)
class RuntimeConfig:
    sample_rate: int = 16_000
    device: Device = "auto"
    max_vram_gb: float = 3.5
    realtime_chunk_ms: int = 20
    seed: int = 42


@dataclass(slots=True)
class AttackConfig:
    stealth: float = 0.8
    strength: float = 0.25
    target_text: str | None = None
    universal: bool = False
    timestamp_s: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    sample_rate: int = 16_000
    duration_s: float = 2.0
    output_dir: str = "results/generated"
    include_baselines: bool = True
