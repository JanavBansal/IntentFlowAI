"""Run top-K holding-period backtest on scored signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.backtest.costs import load_cost_model
from intentflow_ai.config.settings import settings
from intentflow_ai.utils.io import load_price_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest ranked probabilities")
    parser.add_argument("--experiment", default="latest", help="Experiment directory under experiments/")
    parser.add_argument("--top-k", type=int, default=settings.backtest.top_k)
    parser.add_argument("--hold-days", type=int, default=settings.backtest.hold_days)
    parser.add_argument("--slip", type=float, default=settings.backtest.slippage_bps)
    parser.add_argument(
        "--fee",
        default=str(settings.backtest.fee_bps),
        help="Fee in bps or a named cost model (e.g., 'realistic').",
    )
    parser.add_argument(
        "--cost-config",
        default=str(settings.path("config/costs_india.yaml")),
        help="Path to YAML file defining named cost models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = settings.experiments_dir / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    preds_path = exp_dir / "preds.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file missing: {preds_path}")

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    prices = load_price_parquet(allow_fallback=False)

    fee_arg = str(args.fee).strip()
    slippage_bps = float(args.slip)
    try:
        fee_bps = float(fee_arg)
        cost_meta = {
            "name": "manual",
            "components": {"manual_fee_bps": fee_bps},
            "total_bps": fee_bps,
        }
    except ValueError:
        cost_meta = load_cost_model(fee_arg, path=args.cost_config)
        fee_bps = cost_meta["total_bps"]
        slippage_bps = cost_meta.get("slippage_bps", slippage_bps)

    cfg = BacktestConfig(
        top_k=args.top_k,
        hold_days=args.hold_days,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
    )
    result = backtest_signals(preds, prices, cfg)
    if cost_meta:
        result["summary"]["cost_model"] = cost_meta

    equity_path = exp_dir / "bt_equity.csv"
    trades_path = exp_dir / "bt_trades.csv"
    summary_path = exp_dir / "bt_summary.json"

    result["equity_curve"].to_csv(equity_path, header=True)
    result["trades"].to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    print("Backtest summary:", json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
