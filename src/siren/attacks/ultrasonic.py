from __future__ import annotations

import numpy as np

from siren.attacks.base import AttackModule, AttackResult


class UltrasonicAttackSimulator(AttackModule):
    """Generates ultrasonic carriers for detector evaluation and stress testing."""

    name = "ultrasonic"

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        carrier_hz: float = 21_500.0,
        modulation_hz: float = 70.0,
        **kwargs,
    ) -> AttackResult:
        t = np.arange(len(audio), dtype=np.float32) / sample_rate
        carrier = np.sin(2 * np.pi * carrier_hz * t) * (0.5 + 0.5 * np.sin(2 * np.pi * modulation_hz * t))
        out = np.asarray(audio, dtype=np.float32) + 0.08 * carrier
        return AttackResult(
            audio=np.clip(out, -1.0, 1.0),
            metadata={"carrier_hz": carrier_hz, "modulation_hz": modulation_hz},
        )
