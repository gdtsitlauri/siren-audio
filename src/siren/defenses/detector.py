from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from siren.features import band_energy, ultrasonic_energy_ratio
from siren.forensics.training_free import ForensicsReport, TrainingFreeDetector


@dataclass(slots=True)
class DefenseReport:
    blocked: bool
    risk_score: float
    alerts: list[str]
    components: dict[str, float]
    forensics: ForensicsReport


class DefenseSuite:
    def __init__(self) -> None:
        self.forensics = TrainingFreeDetector()

    def analyze(self, audio: np.ndarray, sample_rate: int) -> DefenseReport:
        forensic = self.forensics.analyze(audio, sample_rate)
        ultrasonic = min(1.0, ultrasonic_energy_ratio(audio, sample_rate) * 25.0)
        music_hidden = min(1.0, band_energy(audio, sample_rate, 1_600.0, 3_600.0) / 20.0)
        replay_like = min(1.0, band_energy(audio, sample_rate, 0.0, 500.0) / 30.0)
        components = {
            "forensics_fake_score": forensic.score,
            "ultrasonic_risk": ultrasonic,
            "music_hidden_risk": music_hidden,
            "speaker_spoof_risk": replay_like,
        }
        risk = float(np.mean(list(components.values())))
        alerts = [name for name, value in components.items() if value >= 0.55]
        return DefenseReport(
            blocked=risk >= 0.5 or len(alerts) >= 2,
            risk_score=risk,
            alerts=alerts,
            components=components,
            forensics=forensic,
        )
