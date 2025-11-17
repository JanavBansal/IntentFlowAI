"""Run comprehensive trading metrics analysis with improved metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.modeling.trading_metrics import (
    compare_in_sample_vs_out_of_sample,
    compute_comprehensive_trading_metrics,
    compute_decile_ic,
    compute_sharpe_by_decile,
)
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive trading metrics analysis")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name under experiments/ (e.g., v_universe_sanity)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML config (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load experiment config if provided
    exp_cfg = None
    if args.config:
        exp_cfg = load_experiment_config(args.config)
        cfg = apply_experiment_overrides(Settings(), exp_cfg)
        import intentflow_ai.config.settings as settings_module

        settings_module.settings = cfg
    else:
        cfg = settings

    experiment_name = args.experiment
    exp_dir = cfg.experiments_dir / experiment_name

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    logger.info(f"Running comprehensive metrics analysis for experiment: {experiment_name}")

    # Load training data
    train_path = exp_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run training first.")

    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded training data: {len(train_df)} rows")

    # Load predictions
    preds_path = exp_dir / "preds.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found at {preds_path}. Run training first.")

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    logger.info(f"Loaded predictions: {len(preds)} rows")

    # Load trades if available
    trades_path = exp_dir / "bt_trades.csv"
    trades_df = None
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path, parse_dates=["date_in"])
        logger.info(f"Loaded trades: {len(trades_df)} trades")

    # Load equity curve if available
    equity_path = exp_dir / "bt_equity.csv"
    equity_curve = None
    if equity_path.exists():
        equity_df = pd.read_csv(equity_path, parse_dates=["date"])
        equity_curve = equity_df.set_index("date")["equity"]
        logger.info(f"Loaded equity curve: {len(equity_curve)} points")

    # Get train/test splits from training data
    if "split" in train_df.columns:
        train_mask = train_df["split"] == "train"
        test_mask = train_df["split"] == "test"
    else:
        # Fallback: use date-based split
        if "date" in train_df.columns:
            train_df["date"] = pd.to_datetime(train_df["date"])
            test_start = pd.to_datetime(cfg.test_start) if hasattr(cfg, "test_start") else train_df["date"].max() - pd.Timedelta(days=180)
            train_mask = train_df["date"] < test_start
            test_mask = train_df["date"] >= test_start
        else:
            logger.warning("Cannot determine train/test splits, using all data as test")
            train_mask = pd.Series(False, index=train_df.index)
            test_mask = pd.Series(True, index=train_df.index)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TRADING METRICS ANALYSIS")
    print("=" * 80)

    # Tier 1: Decision Metrics (from trades)
    if trades_df is not None and not trades_df.empty:
        print("\n" + "-" * 80)
        print("TIER 1: DECISION METRICS (From Trades)")
        print("-" * 80)

        # Compute total return from equity curve
        total_return = None
        if equity_curve is not None and not equity_curve.empty:
            total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

        trading_metrics = compute_comprehensive_trading_metrics(
            trades_df,
            ret_col="net_ret",
            equity_curve=equity_curve,
            total_return=total_return,
        )

        print(f"Expected Value per Trade: {trading_metrics.expected_value:.4f} ({trading_metrics.expected_value*100:.2f}%)")
        print(f"  - Hit Rate: {trading_metrics.hit_rate:.2%}")
        print(f"  - Avg Win: {trading_metrics.avg_win:.4f} ({trading_metrics.avg_win*100:.2f}%)")
        print(f"  - Avg Loss: {trading_metrics.avg_loss:.4f} ({trading_metrics.avg_loss*100:.2f}%)")
        print(f"  - Win Count: {trading_metrics.win_count}")
        print(f"  - Loss Count: {trading_metrics.loss_count}")
        print(f"  - Total Trades: {trading_metrics.total_trades}")

        print(f"\nProfit Factor: {trading_metrics.profit_factor:.2f}")
        if trading_metrics.profit_factor < 1.0:
            print("  ❌ Strategy loses money (PF < 1.0)")
        elif trading_metrics.profit_factor < 1.5:
            print("  ⚠️  Strategy is marginal (PF < 1.5)")
        else:
            print("  ✅ Strategy is profitable (PF >= 1.5)")

        print(f"\nHit Rate: {trading_metrics.hit_rate:.2%}")
        if trading_metrics.hit_rate < 0.55:
            print("  ❌ Hit rate too low (< 55%)")
        else:
            print("  ✅ Hit rate acceptable (>= 55%)")

        # Decision gate
        print("\n" + "=" * 80)
        print("DECISION GATE")
        print("=" * 80)
        if trading_metrics.expected_value < 0.001:
            print("❌ STOP TRADING: Expected Value < 0.1% (after costs)")
        elif trading_metrics.profit_factor < 1.0:
            print("❌ STOP TRADING: Profit Factor < 1.0 (strategy loses money)")
        elif trading_metrics.hit_rate < 0.55:
            print("❌ STOP TRADING: Hit Rate < 55% (signal too weak)")
        else:
            print("✅ DECISION METRICS PASS: Strategy is tradeable")

    # Tier 2: Quality Metrics (from predictions)
    print("\n" + "-" * 80)
    print("TIER 2: QUALITY METRICS (Signal Quality)")
    print("-" * 80)

    # Align predictions with training data
    if "excess_fwd" in train_df.columns and "proba" in preds.columns:
        # Ensure date columns are datetime
        train_df["date"] = pd.to_datetime(train_df["date"])
        preds["date"] = pd.to_datetime(preds["date"])
        
        # Merge predictions with training data
        merged = train_df.merge(
            preds[["date", "ticker", "proba"]],
            on=["date", "ticker"],
            how="inner",  # Use inner join to ensure matches
        )

        logger.info(f"Merged data: {len(merged)} rows, columns: {list(merged.columns)}")

        # Check if merge worked
        if "proba" not in merged.columns:
            logger.warning("Merge failed - 'proba' column not found. Using predictions directly.")
            # Use predictions directly instead
            preds_with_date = preds.copy()
            preds_with_date["date"] = pd.to_datetime(preds_with_date["date"])
            if "excess_fwd" in preds_with_date.columns:
                # Use predictions as both train and test (since we don't have split info)
                train_data = preds_with_date.dropna(subset=["proba", "excess_fwd"])
                test_data = preds_with_date.dropna(subset=["proba", "excess_fwd"])
            else:
                logger.warning("Cannot compute quality metrics - missing 'excess_fwd' in predictions")
                train_data = pd.DataFrame()
                test_data = pd.DataFrame()
        else:
            # Recompute masks after merge
            if "split" in merged.columns:
                train_mask_merged = merged["split"] == "train"
                test_mask_merged = merged["split"] == "test"
            else:
                # Fallback: use date-based split
                if "date" in merged.columns:
                    test_start = pd.to_datetime(cfg.test_start) if hasattr(cfg, "test_start") else merged["date"].max() - pd.Timedelta(days=180)
                    train_mask_merged = merged["date"] < test_start
                    test_mask_merged = merged["date"] >= test_start
                else:
                    train_mask_merged = pd.Series(False, index=merged.index)
                    test_mask_merged = pd.Series(True, index=merged.index)

            train_data = merged[train_mask_merged].dropna(subset=["proba", "excess_fwd"])
            test_data = merged[test_mask_merged].dropna(subset=["proba", "excess_fwd"])

        if not train_data.empty and not test_data.empty:
            # In-sample vs Out-of-sample comparison
            comparison = compare_in_sample_vs_out_of_sample(
                train_data["proba"],
                train_data["excess_fwd"],
                test_data["proba"],
                test_data["excess_fwd"],
            )

            print("\nIn-Sample vs Out-of-Sample IC Comparison:")
            print(comparison.to_string(index=False))

            # Decile analysis on test set
            test_decile_ic, test_decile_stats = compute_decile_ic(test_data["proba"], test_data["excess_fwd"])
            test_sharpe_by_decile = compute_sharpe_by_decile(test_data["proba"], test_data["excess_fwd"])

            print(f"\nTest Set Decile IC: {test_decile_ic:.4f}")
            print("\nDecile Statistics (Test Set):")
            print(test_decile_stats.to_string(index=False))

            print("\nSharpe by Decile (Test Set):")
            for i, sharpe in enumerate(test_sharpe_by_decile):
                decile_label = f"Decile {i+1} (Lowest)" if i == 0 else f"Decile {i+1} (Highest)" if i == 9 else f"Decile {i+1}"
                print(f"  {decile_label:20s}: {sharpe:7.4f}")

            # Check if deciles are monotonic
            is_monotonic = all(
                test_sharpe_by_decile[i] <= test_sharpe_by_decile[i + 1]
                for i in range(len(test_sharpe_by_decile) - 1)
                if not (pd.isna(test_sharpe_by_decile[i]) or pd.isna(test_sharpe_by_decile[i + 1]))
            )
            if is_monotonic:
                print("\n✅ Deciles are monotonic (signal strength predicts returns)")
            else:
                print("\n❌ Deciles are NOT monotonic (signal may not be predictive)")

            # Save results
            comparison_path = exp_dir / "in_sample_vs_out_of_sample_ic.csv"
            comparison.to_csv(comparison_path, index=False)
            logger.info(f"Saved IC comparison to {comparison_path}")

            decile_stats_path = exp_dir / "decile_analysis_test.csv"
            test_decile_stats.to_csv(decile_stats_path, index=False)
            logger.info(f"Saved decile analysis to {decile_stats_path}")

    # Tier 3: Risk Metrics
    if equity_curve is not None and not equity_curve.empty:
        print("\n" + "-" * 80)
        print("TIER 3: RISK METRICS")
        print("-" * 80)

        returns = equity_curve.pct_change().dropna()
        roll_max = equity_curve.cummax()
        dd = (equity_curve / roll_max) - 1.0
        max_dd = float(dd.min())

        mean_return = float(returns.mean())
        std_return = float(returns.std())
        sharpe = mean_return / (std_return + 1e-9) * np.sqrt(252) if std_return > 0 else 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = float(downside_returns.std()) if len(downside_returns) > 0 else 0.0
        sortino = mean_return / (downside_std + 1e-9) * np.sqrt(252) if downside_std > 0 else 0.0

        # Calmar ratio
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0

        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Sortino Ratio: {sortino:.2f}")
        print(f"Calmar Ratio: {calmar:.2f}")

        if max_dd < -0.25:
            print("  ❌ Max DD too high (< -25%)")
        else:
            print("  ✅ Max DD acceptable (>= -25%)")

    # Save comprehensive summary
    summary = {
        "experiment": experiment_name,
        "tier_1_decision_metrics": {
            "expected_value": float(trading_metrics.expected_value) if trades_df is not None else None,
            "profit_factor": float(trading_metrics.profit_factor) if trades_df is not None else None,
            "hit_rate": float(trading_metrics.hit_rate) if trades_df is not None else None,
        },
        "tier_2_quality_metrics": {
            "test_return_ic": float(comparison[comparison["metric"] == "return_ic"]["out_of_sample"].iloc[0])
            if not comparison.empty
            else None,
            "test_decile_ic": float(test_decile_ic) if "test_decile_ic" in locals() else None,
        },
        "tier_3_risk_metrics": {
            "max_drawdown": float(max_dd) if equity_curve is not None else None,
            "sharpe": float(sharpe) if equity_curve is not None else None,
            "sortino": float(sortino) if equity_curve is not None else None,
            "calmar": float(calmar) if equity_curve is not None else None,
        },
    }

    summary_path = exp_dir / "comprehensive_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved comprehensive metrics summary to {summary_path}")

    print("\n" + "=" * 80)
    print("Comprehensive Metrics Analysis Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {exp_dir}")


if __name__ == "__main__":
    import numpy as np

    main()

