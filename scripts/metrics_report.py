"""Generate Markdown summary of experiment metrics and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Markdown metrics report for an experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment directory under experiments/")
    return parser.parse_args()


def fmt(value) -> str:
    if value is None:
        return "nan"
    try:
        if isinstance(value, float):
            if value != value:
                return "nan"
            return f"{value:.3f}"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def split_table(metrics: Dict[str, dict]) -> str:
    columns = [
        ("roc_auc", "ROC AUC"),
        ("pr_auc", "PR AUC"),
        ("precision_at_10", "P@10"),
        ("precision_by_day_at_10", "Daily P@10"),
        ("ic", "IC"),
        ("rank_ic", "Rank IC"),
    ]
    header = "| Split | " + " | ".join(col[1] for col in columns) + " |"
    sep = "|" + " --- |" * (len(columns) + 1)
    rows = [header, sep]
    for split in ["train", "valid", "test", "overall"]:
        if split not in metrics:
            continue
        row = f"| {split} | "
        row += " | ".join(fmt(metrics[split].get(field)) for field, _ in columns)
        row += " |"
        rows.append(row)
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    exp_dir = Path("experiments") / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON missing at {metrics_path}")
    metrics = json.loads(metrics_path.read_text())

    report_lines = [f"# Experiment {args.experiment}", "", "## Split Metrics", split_table(metrics), ""]

    plots = metrics.get("plots", {})
    if plots:
        report_lines.append("## Diagnostics")
        for name, rel_path in plots.items():
            display_name = name.replace("_", " ").title()
            report_lines.append(f"![{display_name}]({rel_path})")
        report_lines.append("")

    bt_summary_path = exp_dir / "bt_summary.json"
    if bt_summary_path.exists():
        bt_summary = json.loads(bt_summary_path.read_text())
        report_lines.append("## Backtest Summary")
        for key, value in bt_summary.items():
            report_lines.append(f"- **{key}**: {fmt(value)}")
        report_lines.append("")

    report_path = exp_dir / "metrics_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote metrics report to {report_path}")


if __name__ == "__main__":
    main()
