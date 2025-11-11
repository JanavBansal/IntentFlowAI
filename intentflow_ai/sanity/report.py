"""Markdown report assembly for sanity runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from intentflow_ai.sanity.cost_sweep import CostSweepResult
from intentflow_ai.sanity.leakage_tests import NullLabelResult


@dataclass
class SanityReportBuilder:
    experiment: str
    output_dir: Path

    def build(
        self,
        *,
        metrics: Mapping[str, object],
        ticker_count: int,
        null_result: NullLabelResult,
        cost_results: Sequence[CostSweepResult],
        plots: Iterable[str],
        headline: Mapping[str, float],
    ) -> Path:
        """Persist a Markdown summary that captures sanity highlights."""

        lines = [
            f"# Sanity Report — {self.experiment}",
            "",
            "## Headline Metrics",
        ]
        for key, value in headline.items():
            lines.append(f"- **{key}**: {value:.3f}")
        lines.extend(
            [
                "",
                f"- Training tickers: **{ticker_count}**",
                "",
                "## Null-label Backtest",
                f"- Sharpe: {null_result.sharpe:.3f}",
                f"- IC: {null_result.ic:.3f}",
                f"- Rank IC: {null_result.rank_ic:.3f}",
                "",
                "## Cost Sweep (per-side bps)",
                "",
                "| Model | Total bps | Sharpe | CAGR | MaxDD |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for result in cost_results:
            lines.append(
                f"| {result.fee_label} | {result.total_bps:.1f} | "
                f"{result.sharpe:.3f} | {result.cagr:.3f} | {result.max_dd:.3f} |"
            )
        lines.extend(
            [
                "",
                "## Diagnostics",
            ]
        )
        for rel_path in plots:
            lines.append(f"![{Path(rel_path).stem}]({rel_path})")
        lines.append("")

        import json

        lines.append("## Raw Metrics JSON")
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2, default=str))
        lines.append("```")

        report_path = self.output_dir / "metrics_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path
