from __future__ import annotations

import math

import numpy as np
from scipy.signal import get_window, hilbert, stft

from .audio import frame_audio


def spectral_flux(audio: np.ndarray, sample_rate: int, frame_ms: float = 32.0) -> float:
    frame = max(64, int(sample_rate * frame_ms / 1000.0))
    frame = min(frame, max(2, len(audio)))
    hop = max(1, min(frame - 1, frame // 2))
    _, _, zxx = stft(audio, fs=sample_rate, nperseg=frame, noverlap=frame - hop, boundary=None)
    mag = np.abs(zxx)
    if mag.shape[1] < 2:
        return 0.0
    diff = np.diff(mag, axis=1)
    return float(np.mean(np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=0))))


def phase_coherence(audio: np.ndarray) -> float:
    analytic = hilbert(audio)
    phase = np.unwrap(np.angle(analytic))
    if len(phase) < 3:
        return 1.0
    curvature = np.diff(phase, n=2)
    return float(1.0 / (1.0 + np.std(curvature)))


def zero_crossing_rate(audio: np.ndarray) -> float:
    if len(audio) < 2:
        return 0.0
    return float(np.mean(np.signbit(audio[1:]) != np.signbit(audio[:-1])))


def rms_envelope(audio: np.ndarray, sample_rate: int, frame_ms: float = 25.0) -> np.ndarray:
    frame = max(32, int(sample_rate * frame_ms / 1000.0))
    hop = max(16, frame // 2)
    frames = frame_audio(audio, frame, hop)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-8)


def pitch_proxy(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(128, int(sample_rate * 0.04))
    hop = max(64, int(sample_rate * 0.01))
    frames = frame_audio(audio, frame, hop)
    pitches = []
    for windowed in frames * get_window("hann", frame, fftbins=True):
        corr = np.correlate(windowed, windowed, mode="full")[frame - 1 :]
        min_lag = max(1, sample_rate // 400)
        max_lag = min(len(corr) - 1, sample_rate // 60)
        if max_lag <= min_lag:
            pitches.append(0.0)
            continue
        segment = corr[min_lag:max_lag]
        lag = int(np.argmax(segment) + min_lag)
        pitches.append(sample_rate / lag if lag else 0.0)
    return np.asarray(pitches, dtype=np.float32)


def cepstral_consistency(audio: np.ndarray, sample_rate: int) -> float:
    frame = max(128, int(sample_rate * 0.025))
    hop = max(64, int(sample_rate * 0.01))
    frames = frame_audio(audio, frame, hop)
    coeffs = []
    for f in frames:
        spectrum = np.abs(np.fft.rfft(f * np.hanning(len(f)))) + 1e-6
        cepstrum = np.fft.irfft(np.log(spectrum))
        coeffs.append(cepstrum[:13])
    if len(coeffs) < 2:
        return 1.0
    coeffs_arr = np.stack(coeffs)
    deltas = np.diff(coeffs_arr, axis=0)
    return float(1.0 / (1.0 + np.mean(np.abs(deltas))))


def smoothness(signal: np.ndarray) -> float:
    if len(signal) < 3:
        return 1.0
    return float(1.0 / (1.0 + np.mean(np.abs(np.diff(signal, n=2)))))


def pause_pattern_score(audio: np.ndarray, sample_rate: int) -> float:
    env = rms_envelope(audio, sample_rate)
    threshold = max(1e-4, np.quantile(env, 0.2))
    pauses = env < threshold
    if len(pauses) < 2:
        return 1.0
    lengths = []
    count = 0
    for p in pauses:
        if p:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    if len(lengths) < 2:
        return 0.8
    return float(1.0 / (1.0 + np.std(lengths) / (np.mean(lengths) + 1e-6)))


def ultrasonic_energy_ratio(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 20_000.0) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    total = float(np.sum(spectrum) + 1e-8)
    mask = freqs >= cutoff_hz
    return float(np.sum(spectrum[mask]) / total) if np.any(mask) else 0.0


def band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.mean(spectrum[mask])) if np.any(mask) else 0.0


def normalized_entropy(values: np.ndarray) -> float:
    values = np.abs(values.astype(np.float64)) + 1e-12
    probs = values / np.sum(values)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / math.log2(len(probs))) if len(probs) > 1 else 0.0
