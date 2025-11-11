"""Run top-K holding-period backtest on scored signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.config.settings import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest ranked probabilities")
    parser.add_argument("--experiment", default="latest", help="Experiment directory under experiments/")
    parser.add_argument("--top-k", type=int, default=settings.backtest.top_k)
    parser.add_argument("--hold-days", type=int, default=settings.backtest.hold_days)
    parser.add_argument("--slip", type=float, default=settings.backtest.slippage_bps)
    parser.add_argument("--fee", type=float, default=settings.backtest.fee_bps)
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
    prices_path = settings.data_dir / "raw" / "price_confirmation" / "data.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Price parquet missing: {prices_path}")
    prices = pd.read_parquet(prices_path)

    cfg = BacktestConfig(
        top_k=args.top_k,
        hold_days=args.hold_days,
        slippage_bps=args.slip,
        fee_bps=args.fee,
    )
    result = backtest_signals(preds, prices, cfg)

    equity_path = exp_dir / "bt_equity.csv"
    trades_path = exp_dir / "bt_trades.csv"
    summary_path = exp_dir / "bt_summary.json"

    result["equity_curve"].to_csv(equity_path, header=True)
    result["trades"].to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    print("Backtest summary:", json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
