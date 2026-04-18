from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AttackResult:
    audio: np.ndarray
    metadata: dict[str, float | str | bool]


class AttackModule:
    name = "attack"

    def generate(self, audio: np.ndarray, sample_rate: int, **kwargs) -> AttackResult:
        raise NotImplementedError
