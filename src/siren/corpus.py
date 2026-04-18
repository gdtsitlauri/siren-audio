from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from siren.audio import load_audio, resample_linear
from siren.baselines import lcnn_proxy_score, rawnet2_proxy_score, vanilla_spoof_score
from siren.datasets import AudioRecord, load_dataset_records, load_records_from_manifest, scan_audio_records
from siren.defenses.detector import DefenseSuite
from siren.evaluation import eer, roc_auc_score
from siren.reporting import export_generic_json, export_metrics_csv


@dataclass(slots=True)
class CorpusPrediction:
    path: str
    dataset: str
    split: str
    label: str
    speaker_id: str
    utterance_id: str
    attack_id: str
    detector_score: float
    vanilla_score: float
    lcnn_proxy_score: float
    rawnet2_proxy_score: float


def resolve_records(
    dataset_root: str = "",
    manifest_path: str = "",
    dataset_name: str = "custom",
    protocol_path: str = "",
) -> list[AudioRecord]:
    if dataset_name.lower() in {"asvspoof2019", "fakeavceleb", "librispeech"} and dataset_root:
        return load_dataset_records(dataset_name=dataset_name, dataset_root=dataset_root, protocol_path=protocol_path)
    if manifest_path:
        return load_records_from_manifest(manifest_path, dataset=dataset_name or "manifest")
    if dataset_root:
        return scan_audio_records(dataset_root, dataset=dataset_name or "custom")
    return []


def score_records(records: list[AudioRecord], target_sample_rate: int = 16_000) -> list[CorpusPrediction]:
    suite = DefenseSuite()
    predictions: list[CorpusPrediction] = []
    for record in records:
        audio, sr = load_audio(record.path)
        if sr != target_sample_rate:
            audio = resample_linear(audio, sr, target_sample_rate)
            sr = target_sample_rate
        report = suite.analyze(audio, sr)
        predictions.append(
            CorpusPrediction(
                path=record.path,
                dataset=record.dataset,
                split=record.split,
                label=record.label,
                speaker_id=record.speaker_id,
                utterance_id=record.utterance_id,
                attack_id=record.attack_id,
                detector_score=report.forensics.score,
                vanilla_score=vanilla_spoof_score(audio, sr),
                lcnn_proxy_score=lcnn_proxy_score(audio, sr),
                rawnet2_proxy_score=rawnet2_proxy_score(audio, sr),
            )
        )
    return predictions


def summarize_predictions(predictions: list[CorpusPrediction]) -> dict[str, dict[str, float]]:
    labels = [1 if row.label == "synthetic" else 0 for row in predictions]
    metrics: dict[str, dict[str, float]] = {}
    score_map = {
        "training_free": [row.detector_score for row in predictions],
        "vanilla_baseline": [row.vanilla_score for row in predictions],
        "lcnn_proxy": [row.lcnn_proxy_score for row in predictions],
        "rawnet2_proxy": [row.rawnet2_proxy_score for row in predictions],
    }
    for name, scores in score_map.items():
        metrics[name] = {
            "auc": roc_auc_score(labels, scores),
            "eer": eer(labels, scores),
            "count": float(len(scores)),
        }
    return metrics


def export_predictions(predictions: list[CorpusPrediction], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "corpus_predictions.csv"
    summary_path = out / "corpus_summary.json"
    metrics_path = out / "corpus_summary.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "path",
                "dataset",
                "split",
                "label",
                "speaker_id",
                "utterance_id",
                "attack_id",
                "detector_score",
                "vanilla_score",
                "lcnn_proxy_score",
                "rawnet2_proxy_score",
            ],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(asdict(row))
    summary = summarize_predictions(predictions)
    export_generic_json(summary, summary_path)
    export_metrics_csv(summary, metrics_path)
    return {
        "predictions_csv": str(rows_path),
        "summary_json": str(summary_path),
        "summary_csv": str(metrics_path),
    }
