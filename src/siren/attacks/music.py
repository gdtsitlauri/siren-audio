from __future__ import annotations

import numpy as np

from siren.attacks.base import AttackModule, AttackResult
from siren.config import AttackConfig


class MusicHiddenCommandAttack(AttackModule):
    """Embeds low-energy modulation into a music carrier for simulation benchmarks."""

    name = "music_hidden_command"

    def __init__(self, config: AttackConfig | None = None) -> None:
        self.config = config or AttackConfig(timestamp_s=1.0, strength=0.12, stealth=0.9)

    def generate(self, audio: np.ndarray, sample_rate: int, **kwargs) -> AttackResult:
        cfg = self.config
        t = np.arange(len(audio), dtype=np.float32) / sample_rate
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 2.7 * t))
        payload = (
            np.sin(2 * np.pi * 1800.0 * t)
            + 0.5 * np.sin(2 * np.pi * 2400.0 * t + 0.3)
            + 0.25 * np.sin(2 * np.pi * 3200.0 * t + 0.8)
        )
        start = int(max(0.0, cfg.timestamp_s) * sample_rate)
        if start >= len(audio):
            start = max(0, len(audio) // 4)
        end = min(len(audio), start + int(2.0 * sample_rate))
        injected = np.array(audio, copy=True, dtype=np.float32)
        seg_len = max(0, end - start)
        if seg_len == 0:
            return AttackResult(
                audio=injected,
                metadata={
                    "timestamp_s": cfg.timestamp_s,
                    "stealth": cfg.stealth,
                    "strength": cfg.strength,
                },
            )
        window = np.hanning(seg_len) if seg_len > 1 else np.ones(1, dtype=np.float32)
        segment = payload[:seg_len] * envelope[:seg_len] * window
        injected[start:end] += segment * cfg.strength * (1.0 - 0.5 * cfg.stealth)
        return AttackResult(
            audio=np.clip(injected, -1.0, 1.0),
            metadata={
                "timestamp_s": cfg.timestamp_s,
                "stealth": cfg.stealth,
                "strength": cfg.strength,
            },
        )
