from __future__ import annotations

import numpy as np

from siren.attacks.base import AttackModule, AttackResult
from siren.config import AttackConfig


class SirenWaveAttack(AttackModule):
    """Wavelet-inspired perturbation engine for controlled offline evaluation."""

    name = "siren_wave"

    def __init__(self, config: AttackConfig | None = None) -> None:
        self.config = config or AttackConfig()

    def generate(self, audio: np.ndarray, sample_rate: int, **kwargs) -> AttackResult:
        cfg = self.config
        audio = np.asarray(audio, dtype=np.float32)
        even = audio[::2]
        odd = audio[1::2]
        min_len = min(len(even), len(odd))
        even = even[:min_len]
        odd = odd[:min_len]

        approx = (even + odd) * 0.5
        detail = (even - odd) * 0.5
        mask = np.sqrt(np.abs(approx) + 1e-5)
        signed = np.sign(detail + 1e-6)
        perturb = signed * mask * cfg.strength * (1.0 - 0.65 * cfg.stealth)
        detail_adv = detail + perturb

        recon = np.empty(min_len * 2, dtype=np.float32)
        recon[::2] = approx + detail_adv
        recon[1::2] = approx - detail_adv
        if len(audio) % 2:
            recon = np.concatenate([recon, audio[-1:]])

        blended = 0.9 * audio[: len(recon)] + 0.1 * recon
        if len(audio) > len(recon):
            blended = np.concatenate([blended, audio[len(recon) :]])

        return AttackResult(
            audio=np.clip(blended, -1.0, 1.0),
            metadata={
                "stealth": cfg.stealth,
                "strength": cfg.strength,
                "universal": cfg.universal,
                "perturbation_rms": float(np.sqrt(np.mean((blended - audio) ** 2) + 1e-9)),
            },
        )
