from __future__ import annotations

from dataclasses import asdict

from siren.config import AttackConfig, ExperimentConfig, RuntimeConfig


def default_preset() -> dict[str, object]:
    return {
        "runtime": asdict(RuntimeConfig()),
        "attack": asdict(AttackConfig()),
        "experiment": asdict(ExperimentConfig()),
    }


def low_vram_preset() -> dict[str, object]:
    runtime = RuntimeConfig(max_vram_gb=3.5, realtime_chunk_ms=40)
    attack = AttackConfig(stealth=0.85, strength=0.18)
    experiment = ExperimentConfig(sample_rate=16_000, duration_s=1.5)
    return {"runtime": asdict(runtime), "attack": asdict(attack), "experiment": asdict(experiment)}


def realtime_preset() -> dict[str, object]:
    runtime = RuntimeConfig(realtime_chunk_ms=10)
    attack = AttackConfig(stealth=0.9, strength=0.12)
    experiment = ExperimentConfig(duration_s=1.0)
    return {"runtime": asdict(runtime), "attack": asdict(attack), "experiment": asdict(experiment)}


def preset_manifest() -> dict[str, dict[str, object]]:
    return {
        "default": default_preset(),
        "low_vram_gtx1650": low_vram_preset(),
        "realtime_guard": realtime_preset(),
    }
