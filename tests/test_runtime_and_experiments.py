from pathlib import Path

from siren.experiments import export_experiments, run_experiments, summarize_experiments
from siren.runtime import inspect_runtime
from siren.streaming import StreamingDefense


def test_runtime_probe_is_stable() -> None:
    info = inspect_runtime()
    assert info.backend in {"numpy", "torch"}
    assert info.device


def test_experiment_runner_produces_summary(tmp_path: Path) -> None:
    rows = run_experiments()
    assert rows
    summary = summarize_experiments(rows)
    assert "siren_wave" in summary
    exported = export_experiments(rows, tmp_path)
    assert Path(exported["rows_csv"]).exists()
    assert Path(exported["summary_json"]).exists()


def test_streaming_defense_produces_decision() -> None:
    defense = StreamingDefense(sample_rate=16_000, window_ms=100, history_size=3)
    decision = defense.process_chunk([0.0] * 1600)
    assert 0.0 <= decision.average_risk <= 1.0
