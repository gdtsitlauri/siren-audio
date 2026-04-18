import numpy as np

from siren.attacks.ultrasonic import UltrasonicAttackSimulator
from siren.defenses.detector import DefenseSuite
from siren.forensics.training_free import TrainingFreeDetector


def _speech_like_signal(sample_rate: int = 16_000, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t)
    return env * (
        0.45 * np.sin(2 * np.pi * 180.0 * t)
        + 0.18 * np.sin(2 * np.pi * 360.0 * t)
        + 0.08 * np.sin(2 * np.pi * 540.0 * t)
    )


def test_training_free_detector_returns_breakdown() -> None:
    audio = _speech_like_signal()
    report = TrainingFreeDetector().analyze(audio, 16_000)
    assert 0.0 <= report.score <= 1.0
    assert report.label in {"natural-like", "synthetic-like"}
    assert "phase_coherence" in report.breakdown


def test_defense_suite_flags_ultrasonic_energy() -> None:
    clean = _speech_like_signal(sample_rate=48_000)
    attacked = UltrasonicAttackSimulator().generate(clean, 48_000).audio
    report = DefenseSuite().analyze(attacked, 48_000)
    assert report.components["ultrasonic_risk"] > 0.0
