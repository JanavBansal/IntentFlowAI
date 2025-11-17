"""Run feature audit to analyze features by regime and identify issues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.backtest.filters import RiskFilterConfig
from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.features import (
    FeatureAuditConfig,
    FeatureEngineer,
    audit_feature_stability,
    audit_features_by_regime,
    identify_feature_clusters,
)
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature audit by regime")
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

    logger.info(f"Running feature audit for experiment: {experiment_name}")

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

    # Get labels
    if "label" not in train_df.columns:
        raise ValueError("Training data must have 'label' column")
    labels = train_df["label"]

    # Get dates
    if "date" not in train_df.columns:
        raise ValueError("Training data must have 'date' column")
    dates = pd.to_datetime(train_df["date"])

    # Create audit config
    risk_cfg = RiskFilterConfig()
    if exp_cfg:
        risk_filters = exp_cfg.get("risk_filters", {})
        if risk_filters:
            risk_cfg = RiskFilterConfig(
                **{k: v for k, v in risk_filters.items() if k in RiskFilterConfig.__annotations__}
            )

    audit_cfg = FeatureAuditConfig(
        regime_vol_buckets=args.vol_buckets,
        risk_cfg=risk_cfg,
    )

    # Run regime-based audit
    logger.info("Computing feature audit by regime...")
    regime_audit = audit_features_by_regime(
        features,
        labels,
        prices,
        cfg=audit_cfg,
    )

    if not regime_audit.empty:
        regime_audit_path = exp_dir / "feature_audit_regime.csv"
        regime_audit.to_csv(regime_audit_path, index=False)
        logger.info(f"Saved regime audit to {regime_audit_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("FEATURE AUDIT BY REGIME - SUMMARY")
        print("=" * 80)

        # Top features by IC in each regime
        for regime in ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]:
            regime_data = regime_audit[regime_audit["regime_combined"] == regime].copy()
            if regime_data.empty:
                continue

            regime_data = regime_data.sort_values("ic", ascending=False, na_last=True)
            top_features = regime_data.head(10)

            print(f"\n{regime.upper()} - Top 10 Features by IC:")
            print("-" * 80)
            for _, row in top_features.iterrows():
                print(
                    f"  {row['feature']:40s} | IC: {row['ic']:7.4f} | Rank IC: {row['rank_ic']:7.4f} | "
                    f"Samples: {row['sample_count']:6d}"
                )

        # Features that flip sign across regimes
        print("\n" + "=" * 80)
        print("FEATURES WITH REGIME-DEPENDENT BEHAVIOR")
        print("=" * 80)

        feature_regime_ic = regime_audit.groupby("feature")["ic"].agg(["mean", "std", "min", "max"]).reset_index()
        feature_regime_ic = feature_regime_ic.sort_values("std", ascending=False)
        volatile_features = feature_regime_ic[feature_regime_ic["std"] > 0.1].head(20)

        print("\nTop 20 features with highest IC variance across regimes:")
        print("-" * 80)
        for _, row in volatile_features.iterrows():
            print(
                f"  {row['feature']:40s} | Mean IC: {row['mean']:7.4f} | "
                f"Std: {row['std']:7.4f} | Range: [{row['min']:7.4f}, {row['max']:7.4f}]"
            )

    # Run stability audit
    logger.info("Computing feature stability over time...")
    stability_audit = audit_feature_stability(
        features,
        labels,
        dates,
        window_days=252,
        min_samples=50,
    )

    if not stability_audit.empty:
        stability_path = exp_dir / "feature_audit_stability.csv"
        stability_audit.to_csv(stability_path, index=False)
        logger.info(f"Saved stability audit to {stability_path}")

        # Print stability summary
        print("\n" + "=" * 80)
        print("FEATURE STABILITY OVER TIME")
        print("=" * 80)

        stability_summary = (
            stability_audit.groupby("feature")["ic"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("std", ascending=True)
        )

        print("\nMost stable features (lowest IC std dev):")
        print("-" * 80)
        for _, row in stability_summary.head(20).iterrows():
            print(
                f"  {row['feature']:40s} | Mean IC: {row['mean']:7.4f} | "
                f"Std: {row['std']:7.4f} | Windows: {row['count']:4.0f}"
            )

    # Identify feature clusters
    logger.info("Identifying feature clusters...")
    clusters = identify_feature_clusters(
        features,
        correlation_threshold=args.correlation_threshold,
        min_cluster_size=2,
    )

    if clusters:
        clusters_path = exp_dir / "feature_clusters.json"
        import json

        # Convert to serializable format
        clusters_serializable = {k: v for k, v in clusters.items()}
        clusters_path.write_text(json.dumps(clusters_serializable, indent=2))
        logger.info(f"Saved feature clusters to {clusters_path}")

        print("\n" + "=" * 80)
        print("FEATURE CLUSTERS (Highly Correlated Features)")
        print("=" * 80)
        print(f"\nFound {len(clusters)} clusters with correlation > {args.correlation_threshold}:")
        print("-" * 80)
        for rep, cluster_members in clusters.items():
            print(f"\n  Representative: {rep}")
            print(f"  Cluster size: {len(cluster_members)}")
            print(f"  Members: {', '.join(cluster_members)}")

    print("\n" + "=" * 80)
    print("Feature audit complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

