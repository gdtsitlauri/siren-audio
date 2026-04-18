from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from siren.attacks.music import MusicHiddenCommandAttack
from siren.attacks.speaker import SpeakerSpoofSimulator
from siren.attacks.ultrasonic import UltrasonicAttackSimulator
from siren.audio import save_audio


@dataclass(slots=True)
class GeneratedSample:
    path: str
    label: str
    split: str
    generator: str


def _base_speech(seed: int, sample_rate: int, duration_s: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    f0 = 140.0 + 20.0 * np.sin(2 * np.pi * 2.4 * t + 0.1)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    voiced = 0.32 * np.sin(phase) + 0.13 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase)
    envelope = 0.6 + 0.35 * np.sin(2 * np.pi * 1.5 * t)
    breath = rng.normal(0.0, 0.008, size=len(t)).astype(np.float32)
    return (envelope * voiced + breath).astype(np.float32)


def generate_synthetic_corpus(
    output_root: str | Path,
    count_per_class: int = 4,
    sample_rate: int = 16_000,
    duration_s: float = 2.0,
) -> list[GeneratedSample]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    attacks = [
        ("music_hidden", lambda x, sr: MusicHiddenCommandAttack().generate(x, sr).audio),
        ("speaker_spoof", lambda x, sr: SpeakerSpoofSimulator().generate(x, sr, mode="conversion").audio),
        ("ultrasonic", lambda x, sr: UltrasonicAttackSimulator().generate(x, sr).audio),
    ]
    samples: list[GeneratedSample] = []
    for idx in range(count_per_class):
        split = "dev" if idx % 2 == 0 else "test"
        authentic = _base_speech(100 + idx, sample_rate, duration_s)
        authentic_path = root / split / "authentic" / f"sample_{idx:03d}.wav"
        authentic_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio(authentic_path, authentic, sample_rate)
        samples.append(
            GeneratedSample(
                path=str(authentic_path),
                label="authentic",
                split=split,
                generator="base_speech",
            )
        )
        for attack_name, fn in attacks:
            synth = fn(authentic, sample_rate)
            synth_path = root / split / f"synthetic_{attack_name}" / f"sample_{idx:03d}_{attack_name}.wav"
            synth_path.parent.mkdir(parents=True, exist_ok=True)
            save_audio(synth_path, synth, sample_rate)
            samples.append(
                GeneratedSample(
                    path=str(synth_path),
                    label="synthetic",
                    split=split,
                    generator=attack_name,
                )
            )
    return samples


def export_generated_manifest(samples: list[GeneratedSample], manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "dataset", "split", "attack_id"])
        writer.writeheader()
        for item in samples:
            writer.writerow(
                {
                    "path": item.path,
                    "label": item.label,
                    "dataset": "synthetic_siren",
                    "split": item.split,
                    "attack_id": item.generator if item.label == "synthetic" else "",
                }
            )
    return str(path)
