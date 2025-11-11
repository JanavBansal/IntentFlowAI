"""Transaction cost sensitivity sweeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

import pandas as pd

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.backtest.costs import load_cost_model
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CostSweepResult:
    fee_label: str
    total_bps: float
    sharpe: float
    cagr: float
    max_dd: float


def run_cost_sweep(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    base_cfg: BacktestConfig,
    config_path,
    sweep: Iterable[str | float],
) -> List[CostSweepResult]:
    """Run multiple backtests under different per-side cost assumptions."""

    results: List[CostSweepResult] = []
    for entry in sweep:
        if isinstance(entry, str):
            model = load_cost_model(entry, path=config_path)
            total_bps = model["total_bps"]
            label = entry
        else:
            total_bps = float(entry)
            label = f"{total_bps:.1f}bps"
        cfg = replace(base_cfg, fee_bps=total_bps)
        summary = backtest_signals(preds, prices, cfg)["summary"]
        results.append(
            CostSweepResult(
                fee_label=label,
                total_bps=total_bps,
                sharpe=float(summary.get("Sharpe", 0.0)),
                cagr=float(summary.get("CAGR", 0.0)),
                max_dd=float(summary.get("maxDD", 0.0)),
            )
        )
        logger.info(
            "Cost sweep iteration",
            extra={"label": label, "fee_bps": total_bps, "sharpe": summary.get("Sharpe", 0.0)},
        )
    return results
