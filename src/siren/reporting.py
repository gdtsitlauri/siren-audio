from __future__ import annotations

import csv
import json
from pathlib import Path


def export_markdown_table(summary: dict[str, dict[str, float | bool]], output_path: str | Path, title: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["name", "mean_risk", "blocked_rate", "ci_low", "ci_high"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write("| Name | Mean Risk | Blocked Rate | CI Low | CI High |\n")
        fh.write("|---|---:|---:|---:|---:|\n")
        for name, row in summary.items():
            fh.write(
                f"| {name} | {float(row.get('mean_risk', 0.0)):.4f} | {float(row.get('blocked_rate', 0.0)):.4f} | "
                f"{float(row.get('ci_low', 0.0)):.4f} | {float(row.get('ci_high', 0.0)):.4f} |\n"
            )
    return str(path)


def export_generic_json(payload: dict, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return str(path)


def export_metrics_csv(summary: dict[str, dict[str, float]], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "auc", "eer", "count"])
        writer.writeheader()
        for name, row in summary.items():
            writer.writerow(
                {
                    "name": name,
                    "auc": row.get("auc", 0.0),
                    "eer": row.get("eer", 1.0),
                    "count": row.get("count", 0.0),
                }
            )
    return str(path)
