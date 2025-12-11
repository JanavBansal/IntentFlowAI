#!/usr/bin/env python
"""Feature Ablation Study for Modern Market Features.

Tests which of the 25 new modern features actually help or hurt IC.
Uses the existing train.parquet with all features, runs mini-WFO
with subsets of features to measure impact.

Usage:
    python scripts/feature_ablation_study.py --experiment v_universe_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings
from intentflow_ai.modeling.ensemble import MultiAlgoEnsemble
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)

# Modern market feature groups
MODERN_FEATURE_GROUPS = {
    "index_concentration": [
        "modern_market__top10_volume_share",
        "modern_market__hhi_volume", 
        "modern_market__top10_volume_share_chg",
    ],
    "breadth_divergence": [
        "modern_market__breadth_divergence_5d",
        "modern_market__breadth_divergence_20d",
        "modern_market__breadth_divergence_60d",
        "modern_market__advance_decline_ratio",
        "modern_market__pct_above_50dma",
        "modern_market__pct_above_200dma",
    ],
    "momentum_crowding": [
        "modern_market__momentum_crowding",
        "modern_market__momentum_dispersion",
        "modern_market__momentum_reversal_risk",
    ],
    "passive_flow": [
        "modern_market__days_to_month_end_norm",
        "modern_market__is_month_end_week",
        "modern_market__is_month_start",
        "modern_market__is_quarter_end",
        "modern_market__is_expiry_week",
    ],
    "volatility_regime": [
        "modern_market__market_vol_20d",
        "modern_market__vol_percentile_20d",
        "modern_market__market_vol_60d",
        "modern_market__vol_percentile_60d",
        "modern_market__market_vol_252d",
        "modern_market__vol_percentile_252d",
        "modern_market__vol_regime_score",
        "modern_market__vol_trend",
    ],
}


def compute_ic(predictions: np.ndarray, returns: np.ndarray) -> float:
    """Compute Information Coefficient (Spearman correlation)."""
    valid = ~np.isnan(predictions) & ~np.isnan(returns)
    if valid.sum() < 10:
        return np.nan
    return stats.spearmanr(predictions[valid], returns[valid])[0]


def run_ablation_test(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    exclude_cols: List[str],
    test_year: int = 2023,
) -> Dict:
    """Run a mini WFO test excluding certain features."""
    
    # Filter features
    use_cols = [c for c in feature_cols if c not in exclude_cols]
    
    # Split into train (before test_year) and test (test_year)
    train_mask = train_df["date"].dt.year < test_year
    test_mask = train_df["date"].dt.year == test_year
    
    X_train = train_df.loc[train_mask, use_cols].values
    y_train = train_df.loc[train_mask, "label"].values
    X_test = train_df.loc[test_mask, use_cols].values
    y_test = train_df.loc[test_mask, "label"].values
    returns = train_df.loc[test_mask, "excess_fwd"].values
    
    if len(X_train) < 1000 or len(X_test) < 100:
        return {"ic": np.nan, "n_features": len(use_cols)}
    
    # Train simple model (LightGBM only for speed)
    import lightgbm as lgb
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )
    
    # Handle NaNs
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0) 
    y_train = np.nan_to_num(y_train, nan=0)
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    
    ic = compute_ic(preds, returns)
    
    return {
        "ic": ic,
        "n_features": len(use_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--test-year", type=int, default=2023)
    args = parser.parse_args()
    
    exp_dir = settings.experiments_dir / args.experiment
    train_path = exp_dir / "train.parquet"
    
    logger.info(f"Loading data from {train_path}")
    train_df = pd.read_parquet(train_path)
    train_df["date"] = pd.to_datetime(train_df["date"])
    
    # Get all feature columns (those with __)
    all_feature_cols = [c for c in train_df.columns if "__" in c]
    modern_cols = [c for c in all_feature_cols if c.startswith("modern_market__")]
    base_cols = [c for c in all_feature_cols if not c.startswith("modern_market__")]
    
    logger.info(f"Total features: {len(all_feature_cols)}")
    logger.info(f"Modern features: {len(modern_cols)}")
    logger.info(f"Base features: {len(base_cols)}")
    
    results = []
    
    # Baseline: all features
    logger.info("\n" + "=" * 60)
    logger.info("Running baseline (all features)...")
    baseline = run_ablation_test(train_df, all_feature_cols, [], args.test_year)
    results.append({"name": "ALL_FEATURES", **baseline})
    logger.info(f"Baseline IC: {baseline['ic']:.4f}")
    
    # Without ALL modern features
    logger.info("\nWithout ALL modern features...")
    no_modern = run_ablation_test(train_df, all_feature_cols, modern_cols, args.test_year)
    results.append({"name": "NO_MODERN_FEATURES", **no_modern})
    logger.info(f"No modern IC: {no_modern['ic']:.4f} (delta: {no_modern['ic'] - baseline['ic']:+.4f})")
    
    # Test each modern feature group
    logger.info("\n" + "=" * 60)
    logger.info("Testing individual modern feature groups...")
    
    for group_name, group_cols in MODERN_FEATURE_GROUPS.items():
        # Test WITHOUT this group
        exclude = [c for c in group_cols if c in modern_cols]
        result = run_ablation_test(train_df, all_feature_cols, exclude, args.test_year)
        delta = result["ic"] - baseline["ic"]
        
        results.append({
            "name": f"WITHOUT_{group_name.upper()}",
            "excluded_features": len(exclude),
            **result,
        })
        
        impact = "HELPS" if delta < 0 else "HURTS" if delta > 0 else "NEUTRAL"
        logger.info(f"Without {group_name}: IC = {result['ic']:.4f} (delta: {delta:+.4f}) -> {impact}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = exp_dir / "feature_ablation_results.json"
    results_df.to_json(output_path, orient="records", indent=2)
    logger.info(f"\nResults saved to {output_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    modern_delta = no_modern["ic"] - baseline["ic"]
    if modern_delta > 0.005:
        logger.info(f"⚠️  Modern features are HURTING IC by {modern_delta:.4f}")
        logger.info("    Recommendation: DISABLE modern_market block")
    elif modern_delta < -0.005:
        logger.info(f"✅ Modern features are HELPING IC by {abs(modern_delta):.4f}")
    else:
        logger.info(f"   Modern features have NEUTRAL impact ({modern_delta:+.4f})")


if __name__ == "__main__":
    main()
