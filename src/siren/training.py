from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from siren.attacks.core import SirenWaveAttack
from siren.defenses.detector import DefenseSuite


@dataclass(slots=True)
class AdversarialTrainingReport:
    clean_mean_risk: float
    adversarial_mean_risk: float
    robustness_gain: float


def simulate_adversarial_training(audio_batch: list[np.ndarray], sample_rate: int) -> AdversarialTrainingReport:
    suite = DefenseSuite()
    attack = SirenWaveAttack()
    clean_risks = []
    adv_risks = []
    for audio in audio_batch:
        clean_risks.append(suite.analyze(audio, sample_rate).risk_score)
        adv_audio = attack.generate(audio, sample_rate).audio
        adv_risks.append(suite.analyze(adv_audio, sample_rate).risk_score)
    clean_mean = float(np.mean(clean_risks)) if clean_risks else 0.0
    adv_mean = float(np.mean(adv_risks)) if adv_risks else 0.0
    return AdversarialTrainingReport(
        clean_mean_risk=clean_mean,
        adversarial_mean_risk=adv_mean,
        robustness_gain=max(0.0, adv_mean - clean_mean),
    )
