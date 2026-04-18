from pathlib import Path

import numpy as np

from siren.latency import benchmark_latency
from siren.reporting import export_generic_json, export_markdown_table, export_metrics_csv


def test_reporting_exports_files(tmp_path: Path) -> None:
    summary = {"demo": {"mean_risk": 0.1, "blocked_rate": 0.0, "ci_low": 0.05, "ci_high": 0.15}}
    metrics = {"training_free": {"auc": 0.8, "eer": 0.2, "count": 10.0}}
    md = export_markdown_table(summary, tmp_path / "summary.md", "Demo Summary")
    js = export_generic_json(metrics, tmp_path / "summary.json")
    csv = export_metrics_csv(metrics, tmp_path / "summary.csv")
    assert Path(md).exists()
    assert Path(js).exists()
    assert Path(csv).exists()


def test_latency_benchmark_returns_report() -> None:
    t = np.arange(3200, dtype=np.float32) / 16_000
    audio = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    report = benchmark_latency(audio, runs=2)
    assert report.offline_mean_ms >= 0.0
    assert report.streaming_mean_ms >= 0.0
