# Research Bundle

This directory is intended for release-facing synthetic paper-prep artifacts:

- `experiments/experiment_rows.csv`
- `experiments/experiment_summary.json`
- `experiments/experiment_summary.md`
- `ablation/siren_wave_ablation_rows.csv`
- `ablation/siren_wave_ablation_summary.json`
- `paper_tables/synthetic_experiments.tex`
- `paper_tables/ablation.tex`

Populate with:

```bash
PYTHONPATH=src python -m siren.cli research-bundle --output-dir results/research_bundle
```
