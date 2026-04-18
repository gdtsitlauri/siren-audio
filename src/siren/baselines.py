from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from siren.attacks.base import AttackResult
from siren.features import (
    band_energy,
    cepstral_consistency,
    phase_coherence,
    pitch_proxy,
    smoothness,
    spectral_flux,
    zero_crossing_rate,
)


@dataclass(slots=True)
class BaselineResult:
    name: str
    audio: np.ndarray
    metadata: dict[str, float | str]


def fgsm_audio_baseline(audio: np.ndarray, epsilon: float = 0.01) -> BaselineResult:
    x = np.asarray(audio, dtype=np.float32)
    gradient_proxy = np.sign(np.gradient(x))
    adv = np.clip(x + epsilon * gradient_proxy, -1.0, 1.0)
    return BaselineResult(
        name="fgsm_audio",
        audio=adv,
        metadata={"epsilon": float(epsilon), "proxy": "signal_gradient_sign"},
    )


def cw_audio_baseline(audio: np.ndarray, confidence: float = 0.2) -> BaselineResult:
    x = np.asarray(audio, dtype=np.float32)
    smooth = np.convolve(x, np.ones(9) / 9.0, mode="same")
    perturb = np.tanh(4.0 * (smooth - x)) * confidence * 0.1
    adv = np.clip(x + perturb, -1.0, 1.0)
    return BaselineResult(
        name="cw_audio_proxy",
        audio=adv,
        metadata={"confidence": float(confidence), "proxy": "smoothed_tanh_perturbation"},
    )


def replay_baseline(audio: np.ndarray) -> BaselineResult:
    x = np.asarray(audio, dtype=np.float32)
    delayed = np.pad(x[:-120] if len(x) > 120 else x, (120, 0))
    out = np.clip(0.8 * x + 0.2 * delayed, -1.0, 1.0)
    return BaselineResult(name="replay_baseline", audio=out, metadata={"delay_samples": 120.0})


def as_attack_result(result: BaselineResult) -> AttackResult:
    return AttackResult(audio=result.audio, metadata=result.metadata)


def vanilla_spoof_score(audio: np.ndarray, sample_rate: int) -> float:
    zcr = zero_crossing_rate(audio)
    flux = spectral_flux(audio, sample_rate)
    phase = phase_coherence(audio)
    return float(np.clip(0.35 * zcr + 0.15 * flux + 0.5 * (1.0 - phase), 0.0, 1.0))


def lcnn_proxy_score(audio: np.ndarray, sample_rate: int) -> float:
    cep = cepstral_consistency(audio, sample_rate)
    high_band = band_energy(audio, sample_rate, 3_000.0, 8_000.0)
    pitch = pitch_proxy(audio, sample_rate)
    pitch_smooth = smoothness(pitch[pitch > 0]) if np.any(pitch > 0) else 0.0
    score = 0.45 * (1.0 - cep) + 0.3 * min(1.0, high_band / 30.0) + 0.25 * (1.0 - pitch_smooth)
    return float(np.clip(score, 0.0, 1.0))


def rawnet2_proxy_score(audio: np.ndarray, sample_rate: int) -> float:
    del sample_rate
    energy_jitter = np.mean(np.abs(np.diff(np.abs(audio)))) if len(audio) > 1 else 0.0
    clipping = float(np.mean(np.abs(audio) > 0.98))
    score = 0.6 * min(1.0, energy_jitter * 8.0) + 0.4 * min(1.0, clipping * 10.0)
    return float(np.clip(score, 0.0, 1.0))
