#!/usr/bin/env python3
"""
Horizon Sensitivity Analysis

Test different prediction horizons (5, 10, 15, 30, 90 days) to find optimal IC.
Runs a lightweight WFO for each horizon and compares results.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.labels import make_excess_label
from intentflow_ai.modeling.ensemble import MultiAlgoEnsemble
from intentflow_ai.validation.walk_forward import generate_walk_forward_folds, WalkForwardConfig
from intentflow_ai.utils.logging import get_logger
from intentflow_ai.config.settings import settings

logger = get_logger(__name__)

# Horizons to test
HORIZONS = [5, 10, 15, 30, 90]


def run_horizon_test(
    prices_df: pd.DataFrame,
    horizon_days: int,
    num_folds: int = 5,
    output_dir: Path = None
) -> dict:
    """
    Run WFO for a single horizon and return metrics.
    
    Uses reduced folds for speed (5 instead of 25).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING HORIZON: {horizon_days} DAYS")
    logger.info(f"{'='*60}")
    
    # Create labels with this horizon
    logger.info(f"Creating labels with {horizon_days}-day forward return...")
    
    # Scale threshold with horizon (more realistic for longer periods)
    thresh = 0.005 * (horizon_days / 5)  # ~0.5% per 5 days
    
    labeled_df = make_excess_label(
        prices_df.copy(),
        horizon_days=horizon_days,
        thresh=thresh
    )
    
    # Count label distribution
    label_counts = labeled_df["label"].value_counts()
    logger.info(f"Label distribution: {label_counts.to_dict()}")
    logger.info(f"Samples after labeling: {len(labeled_df)}")
    
    # Build features
    logger.info("Engineering features...")
    
    # IMPORTANT: Preserve label and date before feature engineering (which may drop them)
    labels_preserved = labeled_df["label"].copy()
    dates_preserved = pd.to_datetime(labeled_df["date"]).copy()
    indices_preserved = labeled_df.index.copy()
    
    engineer = FeatureEngineer()
    features_df = engineer.build(labeled_df)
    
    # Restore labels and dates using preserved indices
    # features_df may have fewer rows due to rolling computations
    common_idx = features_df.index.intersection(indices_preserved)
    y = labels_preserved.loc[common_idx].astype(int)
    dates = dates_preserved.loc[common_idx]
    features_df = features_df.loc[common_idx]
    
    # Get feature columns (exclude meta columns)
    exclude_cols = {
        "ticker", "date", "label", "sector", "close", "open", "high", "low", 
        "volume", "excess_fwd", "sector_fwd"
    }
    feature_cols = [c for c in features_df.columns 
                   if c not in exclude_cols 
                   and not c.startswith("fwd_ret")]
    
    X = features_df[feature_cols].copy()
    
    # Handle NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    
    # Configure WFO with fewer folds for speed
    wfo_config = WalkForwardConfig(
        train_start="2015-01-01",  # Start later for faster testing
        valid_duration_days=90,
        test_duration_days=90,
        step_days=180,  # Larger steps for fewer folds
        embargo_days=horizon_days + 5,  # Embargo longer than horizon
    )
    
    # Create temporary DataFrame for fold generation using preserved dates
    temp_df = pd.DataFrame({"date": dates})
    
    folds = generate_walk_forward_folds(
        temp_df,
        wfo_config,
        date_col="date"
    )[:num_folds]
    
    logger.info(f"Running {len(folds)} WFO folds...")
    
    results = []
    
    for i, fold in enumerate(folds):
        try:
            train_mask = fold.train_mask
            test_mask = fold.test_mask
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            if len(X_train) < 100 or len(X_test) < 50:
                logger.warning(f"  Fold {i+1}: Skipped (insufficient data)")
                continue
            
            # Train ensemble
            model = MultiAlgoEnsemble()
            model.train(X_train, y_train)
            
            # Predict
            proba = model.predict_proba(X_test)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            # Compute metrics
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, proba)
            
            # IC: Spearman correlation
            proba_series = pd.Series(proba).reset_index(drop=True)
            y_series = pd.Series(y_test.values).reset_index(drop=True)
            ic = proba_series.corr(y_series, method="spearman")
            
            results.append({
                "fold": i + 1,
                "horizon": horizon_days,
                "auc": auc,
                "ic": ic,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            logger.info(f"  Fold {i+1}: AUC={auc:.4f}, IC={ic:.4f}")
            
        except Exception as e:
            logger.warning(f"  Fold {i+1} failed: {e}")
    
    if not results:
        return {"horizon": horizon_days, "error": "All folds failed"}
    
    # Aggregate results
    df = pd.DataFrame(results)
    
    summary = {
        "horizon": horizon_days,
        "mean_auc": df["auc"].mean(),
        "std_auc": df["auc"].std(),
        "mean_ic": df["ic"].mean(),
        "std_ic": df["ic"].std(),
        "min_ic": df["ic"].min(),
        "max_ic": df["ic"].max(),
        "n_folds": len(df),
        "total_test_samples": df["test_samples"].sum()
    }
    
    logger.info(f"\nHORIZON {horizon_days}d SUMMARY:")
    logger.info(f"  AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    logger.info(f"  IC:  {summary['mean_ic']:.4f} ± {summary['std_ic']:.4f}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Horizon Sensitivity Analysis")
    parser.add_argument(
        "--horizons", 
        type=str, 
        default="5,10,15,30,90",
        help="Comma-separated list of horizons to test"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of WFO folds per horizon (default: 5 for speed)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/horizon_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="v_universe_full",
        help="Experiment to load data from (default: v_universe_full)"
    )
    
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("HORIZON SENSITIVITY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Horizons to test: {horizons}")
    logger.info(f"Folds per horizon: {args.folds}")
    
    # Load data from experiment
    exp_dir = settings.experiments_dir / args.experiment
    train_path = exp_dir / "train.parquet"
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.info("Please run WFO first to create train.parquet")
        sys.exit(1)
    
    logger.info(f"\nLoading data from {train_path}...")
    prices_df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(prices_df)} rows, {prices_df['ticker'].nunique()} tickers")
    
    # Run analysis for each horizon
    all_results = []
    
    for horizon in horizons:
        try:
            result = run_horizon_test(
                prices_df=prices_df,
                horizon_days=horizon,
                num_folds=args.folds,
                output_dir=output_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Horizon {horizon} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"horizon": horizon, "error": str(e)})
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "horizon_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Also save as JSON
    with open(output_dir / "horizon_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("HORIZON SENSITIVITY ANALYSIS - FINAL RESULTS")
    print("=" * 70)
    
    print(f"\n{'Horizon':>10} {'Mean AUC':>12} {'Std AUC':>10} {'Mean IC':>12} {'Std IC':>10}")
    print("-" * 60)
    
    best_ic_horizon = None
    best_ic = -999
    
    for r in all_results:
        if "error" in r:
            print(f"{r['horizon']:>10}d {'ERROR':>12} {r.get('error', '')[:30]}")
        else:
            ic = r['mean_ic']
            flag = " ⭐" if ic > best_ic else ""
            if ic > best_ic:
                best_ic = ic
                best_ic_horizon = r['horizon']
            
            print(f"{r['horizon']:>10}d {r['mean_auc']:>12.4f} {r['std_auc']:>10.4f} "
                  f"{r['mean_ic']:>12.4f} {r['std_ic']:>10.4f}{flag}")
    
    print("-" * 60)
    if best_ic_horizon:
        print(f"\n🏆 BEST HORIZON: {best_ic_horizon} days (IC = {best_ic:.4f})")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
