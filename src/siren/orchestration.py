from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from siren.ablation import export_ablation, run_siren_wave_ablation, summarize_ablation
from siren.experiments import export_experiments, run_experiments, summarize_experiments
from siren.paper_tables import render_ablation_table, render_experiment_table


@dataclass(slots=True)
class ResearchRunReport:
    experiment_summary_path: str
    ablation_summary_path: str
    experiment_table_path: str
    ablation_table_path: str


def run_research_bundle(output_dir: str | Path) -> ResearchRunReport:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    experiment_rows = run_experiments()
    experiment_exports = export_experiments(experiment_rows, out / "experiments")

    ablation_rows = run_siren_wave_ablation()
    ablation_exports = export_ablation(ablation_rows, out / "ablation")

    exp_table = render_experiment_table(
        experiment_exports["summary_json"],
        out / "paper_tables" / "synthetic_experiments.tex",
    )
    abl_table = render_ablation_table(
        ablation_exports["summary_json"],
        out / "paper_tables" / "ablation.tex",
    )
    report = ResearchRunReport(
        experiment_summary_path=experiment_exports["summary_json"],
        ablation_summary_path=ablation_exports["summary_json"],
        experiment_table_path=exp_table,
        ablation_table_path=abl_table,
    )
    with (out / "research_bundle.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2)
    return report
