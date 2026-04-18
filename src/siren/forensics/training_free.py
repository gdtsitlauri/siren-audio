from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from siren.features import (
    band_energy,
    cepstral_consistency,
    normalized_entropy,
    pause_pattern_score,
    phase_coherence,
    pitch_proxy,
    smoothness,
    spectral_flux,
    ultrasonic_energy_ratio,
    zero_crossing_rate,
)


@dataclass(slots=True)
class ForensicsReport:
    score: float
    label: str
    breakdown: dict[str, float]


class TrainingFreeDetector:
    """Training-free audio forensics detector with explainable score outputs."""

    def analyze(self, audio: np.ndarray, sample_rate: int) -> ForensicsReport:
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return ForensicsReport(score=0.0, label="invalid", breakdown={"empty": 1.0})

        pitch = pitch_proxy(audio, sample_rate)
        breakdown = {
            "cepstral_consistency": cepstral_consistency(audio, sample_rate),
            "pitch_smoothness": smoothness(pitch[pitch > 0]) if np.any(pitch > 0) else 0.3,
            "phase_coherence": phase_coherence(audio),
            "pause_pattern_score": pause_pattern_score(audio, sample_rate),
            "spectral_flux_regularity": 1.0 / (1.0 + spectral_flux(audio, sample_rate)),
            "zcr_naturalness": 1.0 / (1.0 + abs(zero_crossing_rate(audio) - 0.12) * 3.0),
            "vocoder_artifact_penalty": 1.0
            - min(1.0, band_energy(audio, sample_rate, 3_500.0, 7_500.0) / 25.0),
            "ultrasonic_cleanliness": 1.0 - min(1.0, ultrasonic_energy_ratio(audio, sample_rate) * 20.0),
            "spectral_entropy": normalized_entropy(np.abs(np.fft.rfft(audio))),
        }

        weighted = {
            "cepstral_consistency": 1.2,
            "pitch_smoothness": 1.1,
            "phase_coherence": 1.1,
            "pause_pattern_score": 0.8,
            "spectral_flux_regularity": 1.0,
            "zcr_naturalness": 0.8,
            "vocoder_artifact_penalty": 1.0,
            "ultrasonic_cleanliness": 0.5,
            "spectral_entropy": 0.7,
        }
        denom = sum(weighted.values())
        naturalness = sum(breakdown[k] * weighted[k] for k in breakdown) / denom
        fake_score = float(np.clip(1.0 - naturalness, 0.0, 1.0))
        label = "synthetic-like" if fake_score >= 0.5 else "natural-like"
        return ForensicsReport(score=fake_score, label=label, breakdown=breakdown)
