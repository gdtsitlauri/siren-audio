from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from siren.config import AttackConfig, ExperimentConfig
from siren.defenses.detector import DefenseSuite
from siren.evaluation import confidence_interval, wilcoxon_signed_rank
from siren.experiments import synthetic_speech


@dataclass(slots=True)
class AblationRow:
    seed: int
    variant: str
    risk_score: float
    blocked: bool


def _variant_generate(audio: np.ndarray, variant: str, strength: float, stealth: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    even = x[::2]
    odd = x[1::2]
    min_len = min(len(even), len(odd))
    if min_len == 0:
        return x
    even = even[:min_len]
    odd = odd[:min_len]
    approx = (even + odd) * 0.5
    detail = (even - odd) * 0.5
    if variant == "no_masking":
        mask = np.ones_like(approx)
    else:
        mask = np.sqrt(np.abs(approx) + 1e-5)
    signed = np.sign(detail + 1e-6)
    if variant == "no_detail_shaping":
        perturb = np.full_like(detail, strength * (1.0 - 0.65 * stealth))
    else:
        perturb = signed * mask * strength * (1.0 - 0.65 * stealth)
    detail_adv = detail + perturb
    recon = np.empty(min_len * 2, dtype=np.float32)
    recon[::2] = approx + detail_adv
    recon[1::2] = approx - detail_adv
    if len(x) % 2:
        recon = np.concatenate([recon, x[-1:]])
    if variant == "no_blending":
        blended = recon
    else:
        blended = 0.9 * x[: len(recon)] + 0.1 * recon
    if len(x) > len(recon):
        blended = np.concatenate([blended, x[len(recon) :]])
    return np.clip(blended, -1.0, 1.0)


def run_siren_wave_ablation(config: ExperimentConfig | None = None) -> list[AblationRow]:
    cfg = config or ExperimentConfig()
    defense = DefenseSuite()
    variants = ("full", "no_masking", "no_blending", "no_detail_shaping")
    rows: list[AblationRow] = []
    base = AttackConfig()
    for seed in cfg.seeds:
        clean = synthetic_speech(seed, cfg.sample_rate, cfg.duration_s)
        for variant in variants:
            attacked = _variant_generate(clean, variant, base.strength, base.stealth)
            report = defense.analyze(attacked, cfg.sample_rate)
            rows.append(
                AblationRow(seed=seed, variant=variant, risk_score=report.risk_score, blocked=report.blocked)
            )
    return rows


def summarize_ablation(rows: list[AblationRow]) -> dict[str, dict[str, float | bool]]:
    grouped: dict[str, list[AblationRow]] = {}
    for row in rows:
        grouped.setdefault(row.variant, []).append(row)
    summary: dict[str, dict[str, float | bool]] = {}
    full_scores = [r.risk_score for r in grouped.get("full", [])]
    for variant, items in grouped.items():
        scores = [x.risk_score for x in items]
        ci_low, ci_high = confidence_interval(scores)
        stat = wilcoxon_signed_rank(full_scores, scores) if variant != "full" and full_scores else {
            "n": float(len(scores)), "statistic": 0.0, "pvalue": 1.0, "mean_delta": 0.0
        }
        summary[variant] = {
            "mean_risk": float(np.mean(scores)) if scores else 0.0,
            "blocked_rate": float(np.mean([1.0 if x.blocked else 0.0 for x in items])) if items else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "pvalue_vs_full": float(stat["pvalue"]),
            "mean_delta_vs_full": float(stat["mean_delta"]),
        }
    return summary


def export_ablation(rows: list[AblationRow], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "siren_wave_ablation_rows.csv"
    json_path = out / "siren_wave_ablation_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "variant", "risk_score", "blocked"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summarize_ablation(rows), fh, indent=2)
    return {"rows_csv": str(csv_path), "summary_json": str(json_path)}
