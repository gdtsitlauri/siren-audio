from pathlib import Path

from siren.ablation import export_ablation, run_siren_wave_ablation, summarize_ablation
from siren.orchestration import run_research_bundle
from siren.paper_tables import render_ablation_table, render_experiment_table


def test_ablation_runner_exports(tmp_path: Path) -> None:
    rows = run_siren_wave_ablation()
    assert rows
    summary = summarize_ablation(rows)
    assert "full" in summary
    exported = export_ablation(rows, tmp_path / "ablation")
    assert Path(exported["summary_json"]).exists()


def test_table_renderers_and_bundle(tmp_path: Path) -> None:
    bundle = run_research_bundle(tmp_path / "bundle")
    assert Path(bundle.experiment_table_path).exists()
    assert Path(bundle.ablation_table_path).exists()
    exp_out = render_experiment_table(bundle.experiment_summary_path, tmp_path / "tables" / "exp.tex")
    abl_out = render_ablation_table(bundle.ablation_summary_path, tmp_path / "tables" / "abl.tex")
    assert Path(exp_out).exists()
    assert Path(abl_out).exists()
