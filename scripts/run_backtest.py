"""Run top-K holding-period backtest on scored signals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.backtest.costs import load_cost_model
from intentflow_ai.backtest.filters import MetaFilterConfig, RiskFilterConfig
from intentflow_ai.config.experiments import apply_experiment_overrides, backtest_params_from_experiment, load_experiment_config
from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.utils.io import load_price_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest ranked probabilities")
    parser.add_argument("--experiment", default=None, help="Experiment directory under experiments/")
    parser.add_argument(
        "--config",
        default="config/experiments/v_universe_sanity.yaml",
        help="Experiment YAML describing splits/backtest knobs.",
    )
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
    exp_cfg = load_experiment_config(args.config) if args.config else None
    exp_params = backtest_params_from_experiment(exp_cfg)
    cfg = apply_experiment_overrides(Settings(), exp_cfg)
    import intentflow_ai.config.settings as settings_module

    settings_module.settings = cfg
    experiment_name = args.experiment or (exp_cfg.id if exp_cfg else "latest")
    exp_dir = cfg.experiments_dir / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    preds_path = exp_dir / "preds.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file missing: {preds_path}")

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    prices = load_price_parquet(
        allow_fallback=False,
        start_date=getattr(cfg, "price_start", None),
        end_date=getattr(cfg, "price_end", None),
        cfg=cfg,
    )

    fee_arg = str(exp_params.get("cost_model") or exp_params.get("fee_bps") or args.fee).strip()
    slippage_bps = float(exp_params.get("slippage_bps") or args.slip)
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

    top_k = exp_params.get("top_k") or args.top_k
    hold_days = exp_params.get("hold_days") or args.hold_days
    risk_params = {k: v for k, v in exp_params.get("risk_filters", {}).items() if k in RiskFilterConfig.__annotations__}
    meta_params = {k: v for k, v in exp_params.get("meta_labeling", {}).items() if k in MetaFilterConfig.__annotations__}
    risk_cfg = RiskFilterConfig(**risk_params)
    meta_cfg = MetaFilterConfig(**meta_params)

    cfg = BacktestConfig(
        top_k=top_k,
        hold_days=hold_days,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
        risk=risk_cfg,
        meta=meta_cfg,
    )
    result = backtest_signals(preds, prices, cfg)
    if cost_meta:
        result["summary"]["cost_model"] = cost_meta
    result["summary"]["risk_filters"] = asdict(risk_cfg)
    result["summary"]["meta_filter"] = asdict(meta_cfg)

    equity_path = exp_dir / "equity_curve.csv"
    trades_path = exp_dir / "trades.csv"
    summary_csv_path = exp_dir / "backtest_summary.csv"
    summary_json_path = exp_dir / "bt_summary.json"

    result["equity_curve"].to_csv(equity_path, header=True)
    trades_df = result["trades"]
    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame([result["summary"]]).to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    # Legacy filenames for compatibility
    result["equity_curve"].to_csv(exp_dir / "bt_equity.csv", header=True)
    trades_df.to_csv(exp_dir / "bt_trades.csv", index=False)

    print("Backtest summary:", json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
