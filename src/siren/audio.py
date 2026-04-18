from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), clipped, sample_rate)


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    new_len = max(1, int(round(len(audio) * target_sr / orig_sr)))
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def frame_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(audio) < frame_size:
        pad = np.pad(audio, (0, frame_size - len(audio)))
        return pad[None, :]
    starts = np.arange(0, len(audio) - frame_size + 1, hop_size, dtype=int)
    return np.stack([audio[s : s + frame_size] for s in starts], axis=0)
