from __future__ import annotations

import numpy as np

from siren.attacks.base import AttackModule, AttackResult


class SpeakerSpoofSimulator(AttackModule):
    """Simulates replay, conversion, and synthesis-style spoofing artifacts."""

    name = "speaker_spoof"

    def generate(self, audio: np.ndarray, sample_rate: int, mode: str = "conversion", **kwargs) -> AttackResult:
        x = np.asarray(audio, dtype=np.float32)
        if mode == "replay":
            delayed = np.pad(x[:-160] if len(x) > 160 else x, (160, 0))
            out = 0.78 * x + 0.22 * delayed
        elif mode == "synthesis":
            grain = np.sign(x) * np.sqrt(np.abs(x) + 1e-5)
            out = 0.7 * grain + 0.3 * np.roll(grain, 80)
        else:
            out = 0.82 * x + 0.18 * np.roll(x, 48)
            out = np.tanh(1.15 * out)
        return AttackResult(audio=np.clip(out, -1.0, 1.0), metadata={"mode": mode})
