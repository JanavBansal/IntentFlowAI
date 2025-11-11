"""Run the sanity kit to validate data scope, leakage, and cost realism."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.backtest.core import BacktestConfig
from intentflow_ai.config import Settings
from intentflow_ai.sanity import (
    CostSweepResult,
    DataScopeChecks,
    SanityReportBuilder,
    run_cost_sweep,
    run_null_label_test,
    verify_forward_alignment,
)
from intentflow_ai.utils import time_splits
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity checks on an experiment output.")
    parser.add_argument("--experiment", required=True, help="Experiment directory under experiments/")
    parser.add_argument("--config", default="config/costs_india.yaml", help="Cost-model YAML.")
    parser.add_argument(
        "--cost-sweep",
        nargs="*",
        default=["realistic", "sweep_10bps", "sweep_25bps", "sweep_50bps", "sweep_75bps"],
        help="List of cost model names or explicit bps (floats) for the sweep.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def save_plot(series: pd.Series, path: Path, title: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    settings = Settings()
    exp_dir = Path("experiments") / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    metrics = load_json(exp_dir / "metrics.json")
    preds = pd.read_csv(exp_dir / "preds.csv", parse_dates=["date"])
    summary = load_json(exp_dir / "bt_summary.json")
    trades_path = exp_dir / "bt_trades.csv"
    equity_path = exp_dir / "bt_equity.csv"
    train_frame_path = exp_dir / "train.parquet"
    if not train_frame_path.exists():
        raise FileNotFoundError("Training frame parquet missing; rerun training with latest code.")
    train_frame = pd.read_parquet(train_frame_path)

    train_mask, _, _ = time_splits(
        train_frame,
        valid_start=settings.valid_start,
        test_start=settings.test_start,
    )
    data_scope = DataScopeChecks(min_train_tickers=settings.min_train_tickers)
    snapshot_path = exp_dir / "universe_snapshot.csv"
    ticker_count = data_scope.validate_and_snapshot(train_frame, train_mask, output_path=snapshot_path)

    verify_forward_alignment(
        train_frame,
        price_col="close",
        ticker_col="ticker",
        date_col="date",
        fwd_ret_col=f"fwd_ret_{settings.signal_horizon_days}d",
        horizon_days=settings.signal_horizon_days,
    )

    bundle = load(exp_dir / "lgb.pkl")
    feature_cols = bundle.get("feature_columns", [])
    if not feature_cols:
        raise ValueError("Feature columns missing from model bundle; unable to run sanity tests.")

    price_panel = load_price_parquet(allow_fallback=False)
    base_cfg = BacktestConfig(
        top_k=settings.backtest.top_k,
        hold_days=settings.backtest.hold_days,
        slippage_bps=summary.get("cost_model", {}).get("slippage_bps", settings.backtest.slippage_bps),
        fee_bps=summary.get("cost_model", {}).get("total_bps", settings.backtest.fee_bps),
    )

    null_result = run_null_label_test(
        training_frame=train_frame,
        feature_columns=feature_cols,
        horizon_days=settings.signal_horizon_days,
        seed=settings.lgbm_seed,
        price_panel=price_panel,
        backtest_cfg=base_cfg,
        lgbm_cfg=settings.lgbm,
        output_dir=exp_dir / "sanity",
    )

    sweep_entries: List[str | float] = []
    for entry in args.cost_sweep:
        try:
            sweep_entries.append(float(entry))
        except ValueError:
            sweep_entries.append(entry)
    cost_results = run_cost_sweep(
        preds,
        price_panel,
        base_cfg=base_cfg,
        config_path=args.config,
        sweep=sweep_entries,
    )
    if cost_results:
        sweep_path = exp_dir / "cost_sweep.csv"
        pd.DataFrame([r.__dict__ for r in cost_results]).to_csv(sweep_path, index=False)

    plots_dir = exp_dir / "plots"
    plot_paths: List[str] = [str(Path(rel)) for rel in metrics.get("plots", {}).values()]
    if equity_path.exists():
        equity = pd.read_csv(equity_path, parse_dates=[0], index_col=0).squeeze("columns")
        plot_paths.append(
            str(
                save_plot(equity, plots_dir / "equity_curve.png", "Equity Curve", "Equity (x)").relative_to(exp_dir)
            )
        )
    exposure = float("nan")
    capacity_hint = float("nan")
    if trades_path.exists():
        trades = pd.read_csv(trades_path, parse_dates=["date_in"])
        if not trades.empty:
            avg_positions = trades.groupby("date_in").size().mean()
            exposure = avg_positions / settings.backtest.top_k
            capacity_hint = avg_positions * settings.backtest.hold_days

    daily_hits = (
        preds.sort_values(["date", "proba"], ascending=[True, False])
        .groupby("date")
        .head(settings.backtest.top_k)
        .groupby("date")["label"]
        .mean()
    )
    if not daily_hits.empty:
        plot_paths.append(
            str(
                save_plot(
                    daily_hits, plots_dir / "hit_rate_by_day.png", "Precision@K by Day", "Hit Rate"
                ).relative_to(exp_dir)
            )
        )
    daily_ic = preds.groupby("date").apply(lambda g: g["proba"].corr(g["excess_fwd"]))
    if not daily_ic.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        daily_ic.dropna().hist(ax=ax, bins=20)
        ax.set_title("Daily IC Distribution")
        ax.set_xlabel("Information Coefficient")
        ax.grid(True, alpha=0.3)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ic_path = plots_dir / "ic_distribution.png"
        fig.tight_layout()
        fig.savefig(ic_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(ic_path.relative_to(exp_dir)))

    headline = {
        "Sharpe": float(summary.get("Sharpe", 0.0)),
        "MaxDD": float(summary.get("maxDD", 0.0)),
        "AvgIC": float(metrics.get("overall", {}).get("ic", 0.0)),
    }
    if not pd.isna(exposure):
        headline["AvgExposure"] = exposure
    if not pd.isna(capacity_hint):
        headline["CapacityHint"] = capacity_hint

    report = SanityReportBuilder(args.experiment, exp_dir).build(
        metrics=metrics,
        ticker_count=ticker_count,
        null_result=null_result,
        cost_results=cost_results,
        plots=plot_paths,
        headline=headline,
    )

    print(
        f"Sanity summary -> Sharpe: {headline['Sharpe']:.3f}, "
        f"MaxDD: {headline['MaxDD']:.3f}, AvgIC: {headline['AvgIC']:.3f}"
    )
    print(f"Report: {report}")

    if headline["Sharpe"] > 3 and headline["AvgIC"] < 0.03:
        raise SystemExit("Aborting: Sharpe > 3.0 while AvgIC < 0.03 (potential leakage).")
    if abs(null_result.sharpe) > 0.5:
        raise SystemExit("Aborting: Null-label Sharpe magnitude exceeds 0.5 (should be near zero).")


if __name__ == "__main__":
    main()
