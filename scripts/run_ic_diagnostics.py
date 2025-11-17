"""Run comprehensive IC diagnostics to understand feature predictive power."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.backtest.filters import RiskFilterConfig
from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.features import FeatureEngineer
from intentflow_ai.features.ic_diagnostics import (
    ICDiagnosticsConfig,
    analyze_cross_feature_correlation,
    analyze_ic_by_feature_block,
    analyze_ic_by_regime,
    analyze_rolling_ic_over_time,
    build_orthogonal_factors,
    detect_ic_breakpoints,
)
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IC diagnostics")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name under experiments/ (e.g., v_universe_sanity)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML config (optional, for loading price data)",
    )
    parser.add_argument(
        "--vol-buckets",
        type=float,
        nargs="+",
        default=[0, 33, 67, 100],
        help="Volatility percentile buckets for regime classification (default: 0 33 67 100)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.8,
        help="Correlation threshold for feature clustering (default: 0.8)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=252,
        help="Rolling window size in days for IC analysis (default: 252)",
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

    logger.info(f"Running IC diagnostics for experiment: {experiment_name}")

    # Load training data
    train_path = exp_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run training first.")

    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded training data: {len(train_df)} rows")

    # Load prices for regime classification
    prices = load_price_parquet(
        allow_fallback=True,
        start_date=getattr(cfg, "price_start", None),
        end_date=getattr(cfg, "price_end", None),
        cfg=cfg,
    )
    if prices.empty:
        raise ValueError("Price data is empty, cannot compute regime flags")

    logger.info(f"Loaded price data: {len(prices)} rows")

    # Build features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.build(train_df)
    logger.info(f"Built {len(features.columns)} features")

    # Get labels and dates
    if "label" not in train_df.columns:
        raise ValueError("Training data must have 'label' column")
    labels = train_df["label"]
    dates = pd.to_datetime(train_df["date"])

    # Create diagnostics config
    risk_cfg = RiskFilterConfig()
    if exp_cfg:
        risk_filters = exp_cfg.get("risk_filters", {})
        if risk_filters:
            risk_cfg = RiskFilterConfig(
                **{k: v for k, v in risk_filters.items() if k in RiskFilterConfig.__annotations__}
            )

    diag_cfg = ICDiagnosticsConfig(
        regime_vol_buckets=args.vol_buckets,
        risk_cfg=risk_cfg,
        rolling_window_days=args.rolling_window,
        correlation_threshold=args.correlation_threshold,
    )

    print("\n" + "=" * 80)
    print("IC DIAGNOSTICS - COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # 1. IC Analysis by Feature Block
    logger.info("Analyzing IC by feature block...")
    block_ic = analyze_ic_by_feature_block(features, labels)

    if not block_ic.empty:
        block_ic_path = exp_dir / "ic_by_feature_block.csv"
        block_ic.to_csv(block_ic_path, index=False)
        logger.info(f"Saved block IC analysis to {block_ic_path}")

        print("\n" + "-" * 80)
        print("IC BY FEATURE BLOCK")
        print("-" * 80)
        print(block_ic.to_string(index=False))

    # 2. Rolling IC Over Time
    logger.info("Analyzing rolling IC over time...")
    rolling_ic = analyze_rolling_ic_over_time(
        features,
        labels,
        dates,
        window_days=args.rolling_window,
        min_samples=50,
    )

    if not rolling_ic.empty:
        rolling_ic_path = exp_dir / "ic_rolling_over_time.csv"
        rolling_ic.to_csv(rolling_ic_path, index=False)
        logger.info(f"Saved rolling IC to {rolling_ic_path}")

        # Detect breakpoints
        breakpoints = detect_ic_breakpoints(rolling_ic, ic_drop_threshold=0.05)

        if not breakpoints.empty:
            breakpoints_path = exp_dir / "ic_breakpoints.csv"
            breakpoints.to_csv(breakpoints_path, index=False)
            logger.info(f"Saved IC breakpoints to {breakpoints_path}")

            print("\n" + "-" * 80)
            print("IC BREAKPOINTS (Market Structure Changes)")
            print("-" * 80)
            print(breakpoints.to_string(index=False))

        # Summary statistics
        print("\n" + "-" * 80)
        print("ROLLING IC SUMMARY")
        print("-" * 80)
        print(f"Mean IC over time: {rolling_ic['mean_ic'].mean():.4f}")
        print(f"Std IC over time: {rolling_ic['mean_ic'].std():.4f}")
        print(f"Min IC: {rolling_ic['mean_ic'].min():.4f} (at {rolling_ic.loc[rolling_ic['mean_ic'].idxmin(), 'window_center']})")
        print(f"Max IC: {rolling_ic['mean_ic'].max():.4f} (at {rolling_ic.loc[rolling_ic['mean_ic'].idxmax(), 'window_center']})")

    # 3. IC by Regime
    logger.info("Analyzing IC by market regime...")
    regime_ic = analyze_ic_by_regime(
        features,
        labels,
        prices,
        cfg=diag_cfg,
    )

    if not regime_ic.empty:
        regime_ic_path = exp_dir / "ic_by_regime.csv"
        regime_ic.to_csv(regime_ic_path, index=False)
        logger.info(f"Saved regime IC analysis to {regime_ic_path}")

        print("\n" + "-" * 80)
        print("IC BY MARKET REGIME")
        print("-" * 80)

        # Summary by regime
        regime_summary = (
            regime_ic.groupby("regime_combined")["ic"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        print("\nRegime Summary:")
        print(regime_summary.to_string(index=False))

        # Features that work best in each regime
        print("\nTop 5 Features by IC in Each Regime:")
        for regime in regime_summary["regime_combined"].head(6):
            regime_features = regime_ic[regime_ic["regime_combined"] == regime].sort_values("ic", ascending=False).head(5)
            print(f"\n{regime.upper()}:")
            for _, row in regime_features.iterrows():
                print(f"  {row['feature']:40s} | IC: {row['ic']:7.4f} | Samples: {row['sample_count']:6d}")

    # 4. Cross-Feature Correlation Analysis
    logger.info("Analyzing cross-feature correlations...")
    correlation_analysis = analyze_cross_feature_correlation(
        features,
        correlation_threshold=args.correlation_threshold,
        min_cluster_size=2,
    )

    if not correlation_analysis.empty:
        corr_path = exp_dir / "feature_correlation_clusters.csv"
        correlation_analysis.to_csv(corr_path, index=False)
        logger.info(f"Saved correlation analysis to {corr_path}")

        print("\n" + "-" * 80)
        print("FEATURE CORRELATION CLUSTERS")
        print("-" * 80)
        print(f"Found {len(correlation_analysis)} clusters with correlation > {args.correlation_threshold}")
        print(correlation_analysis.to_string(index=False))

    # 5. Orthogonal Factors
    logger.info("Building orthogonal factors...")
    factor_values, factor_ic = build_orthogonal_factors(features, labels)

    if not factor_ic.empty:
        factor_ic_path = exp_dir / "orthogonal_factors_ic.csv"
        factor_ic.to_csv(factor_ic_path, index=False)
        logger.info(f"Saved orthogonal factors IC to {factor_ic_path}")

        if "max_correlation" in factor_ic.columns:
            print("\n" + "-" * 80)
            print("ORTHOGONAL FACTORS")
            print("-" * 80)
            print(factor_ic.to_string(index=False))

            # Check orthogonality
            high_corr_factors = factor_ic[factor_ic["max_correlation"] > 0.5]
            if not high_corr_factors.empty:
                print("\n⚠️  WARNING: Some factors are highly correlated (not orthogonal):")
                print(high_corr_factors[["factor", "max_correlation"]].to_string(index=False))
            else:
                print("\n✅ Factors are reasonably orthogonal (max correlation < 0.5)")

    # Save summary JSON
    summary = {
        "experiment": experiment_name,
        "total_features": len(features.columns),
        "block_ic_summary": block_ic.to_dict("records") if not block_ic.empty else [],
        "rolling_ic_summary": {
            "mean": float(rolling_ic["mean_ic"].mean()) if not rolling_ic.empty else None,
            "std": float(rolling_ic["mean_ic"].std()) if not rolling_ic.empty else None,
            "min": float(rolling_ic["mean_ic"].min()) if not rolling_ic.empty else None,
            "max": float(rolling_ic["mean_ic"].max()) if not rolling_ic.empty else None,
        },
        "regime_ic_summary": regime_summary.to_dict("records") if not regime_ic.empty else [],
        "correlation_clusters": len(correlation_analysis) if not correlation_analysis.empty else 0,
        "orthogonal_factors": factor_ic.to_dict("records") if not factor_ic.empty else [],
    }

    summary_path = exp_dir / "ic_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved summary to {summary_path}")

    print("\n" + "=" * 80)
    print("IC Diagnostics Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {exp_dir}")


if __name__ == "__main__":
    main()

