from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from siren.attacks.core import SirenWaveAttack
from siren.attacks.music import MusicHiddenCommandAttack
from siren.attacks.speaker import SpeakerSpoofSimulator
from siren.attacks.ultrasonic import UltrasonicAttackSimulator
from siren.baselines import cw_audio_baseline, fgsm_audio_baseline, replay_baseline
from siren.config import ExperimentConfig
from siren.defenses.detector import DefenseSuite
from siren.evaluation import confidence_interval
from siren.reporting import export_markdown_table


@dataclass(slots=True)
class ExperimentRow:
    seed: int
    module: str
    risk_score: float
    blocked: bool


def synthetic_speech(seed: int, sample_rate: int, duration_s: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    f0 = 160.0 + 18.0 * np.sin(2 * np.pi * 1.9 * t) + 8.0 * np.sin(2 * np.pi * 3.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    voiced = 0.35 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.07 * np.sin(3 * phase)
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.2 * t + 0.15)
    breath = rng.normal(0.0, 0.01, size=len(t)).astype(np.float32)
    pauses = ((np.sin(2 * np.pi * 0.8 * t) + 0.15) > -0.6).astype(np.float32)
    return (envelope * voiced * pauses + breath).astype(np.float32)


def _attack_bank() -> list[tuple[str, callable]]:
    return [
        ("siren_wave", lambda x, sr: SirenWaveAttack().generate(x, sr).audio),
        ("music_hidden_command", lambda x, sr: MusicHiddenCommandAttack().generate(x, sr).audio),
        ("speaker_spoof", lambda x, sr: SpeakerSpoofSimulator().generate(x, sr).audio),
        ("ultrasonic", lambda x, sr: UltrasonicAttackSimulator().generate(x, sr).audio),
        ("fgsm_audio", lambda x, sr: fgsm_audio_baseline(x).audio),
        ("cw_audio_proxy", lambda x, sr: cw_audio_baseline(x).audio),
        ("replay_baseline", lambda x, sr: replay_baseline(x).audio),
    ]


def run_experiments(config: ExperimentConfig | None = None) -> list[ExperimentRow]:
    cfg = config or ExperimentConfig()
    defense = DefenseSuite()
    rows: list[ExperimentRow] = []
    for seed in cfg.seeds:
        clean = synthetic_speech(seed, cfg.sample_rate, cfg.duration_s)
        for name, fn in _attack_bank():
            attacked = fn(clean, cfg.sample_rate)
            report = defense.analyze(attacked, cfg.sample_rate)
            rows.append(
                ExperimentRow(
                    seed=seed,
                    module=name,
                    risk_score=report.risk_score,
                    blocked=report.blocked,
                )
            )
    return rows


def summarize_experiments(rows: list[ExperimentRow]) -> dict[str, dict[str, float | bool]]:
    grouped: dict[str, list[ExperimentRow]] = {}
    for row in rows:
        grouped.setdefault(row.module, []).append(row)
    summary: dict[str, dict[str, float | bool]] = {}
    for module, items in grouped.items():
        risks = [x.risk_score for x in items]
        blocked_rate = float(np.mean([1.0 if x.blocked else 0.0 for x in items]))
        ci_low, ci_high = confidence_interval(risks)
        summary[module] = {
            "mean_risk": float(np.mean(risks)),
            "blocked_rate": blocked_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "fully_blocked": blocked_rate >= 0.999,
        }
    return summary


def export_experiments(rows: list[ExperimentRow], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "experiment_rows.csv"
    json_path = out / "experiment_summary.json"
    md_path = out / "experiment_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "module", "risk_score", "blocked"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    summary = summarize_experiments(rows)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    export_markdown_table(summary, md_path, title="Synthetic Experiment Summary")
    return {"rows_csv": str(csv_path), "summary_json": str(json_path), "summary_md": str(md_path)}
