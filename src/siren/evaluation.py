from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from siren.attacks.core import SirenWaveAttack
from siren.attacks.music import MusicHiddenCommandAttack
from siren.attacks.speaker import SpeakerSpoofSimulator
from siren.attacks.ultrasonic import UltrasonicAttackSimulator
from siren.defenses.detector import DefenseSuite


@dataclass(slots=True)
class BenchmarkRow:
    module: str
    score: float
    detected: bool


def evaluate_suite(audio: np.ndarray, sample_rate: int) -> list[BenchmarkRow]:
    defense = DefenseSuite()
    attacks = [
        SirenWaveAttack(),
        MusicHiddenCommandAttack(),
        SpeakerSpoofSimulator(),
        UltrasonicAttackSimulator(),
    ]
    rows: list[BenchmarkRow] = []
    for attack in attacks:
        attacked = attack.generate(audio, sample_rate).audio
        report = defense.analyze(attacked, sample_rate)
        rows.append(BenchmarkRow(module=attack.name, score=report.risk_score, detected=report.blocked))
    return rows


def confidence_interval(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    delta = float(1.96 * np.std(arr, ddof=0) / np.sqrt(len(arr)))
    return mean - delta, mean + delta


def wilcoxon_signed_rank_proxy(values_a: list[float], values_b: list[float]) -> dict[str, float]:
    """Small dependency-free proxy summary for paired comparisons."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        return {"n": 0.0, "mean_delta": 0.0, "positive_fraction": 0.0}
    delta = a - b
    return {
        "n": float(len(delta)),
        "mean_delta": float(np.mean(delta)),
        "positive_fraction": float(np.mean(delta > 0.0)),
    }


def wilcoxon_signed_rank(values_a: list[float], values_b: list[float]) -> dict[str, float]:
    if len(values_a) != len(values_b) or len(values_a) == 0:
        return {"n": 0.0, "statistic": 0.0, "pvalue": 1.0, "mean_delta": 0.0}
    try:
        result = wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)
    except Exception:
        proxy = wilcoxon_signed_rank_proxy(values_a, values_b)
        return {"n": proxy["n"], "statistic": 0.0, "pvalue": 1.0, "mean_delta": proxy["mean_delta"]}
    delta = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    return {
        "n": float(len(delta)),
        "statistic": statistic,
        "pvalue": pvalue,
        "mean_delta": float(np.mean(delta)),
    }


def roc_auc_score(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or not labels:
        return 0.0
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0.0
    for p in positives:
        for n in negatives:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / total) if total else 0.0


def eer(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or not labels:
        return 1.0
    thresholds = sorted(set(scores))
    best = 1.0
    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in scores]
        fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
        tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
        tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        best = min(best, abs(fpr - fnr) + (fpr + fnr) / 2.0)
    return float(best)
