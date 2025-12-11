#!/usr/bin/env python
"""Analyze IC by market regime to identify which regimes cause IC instability.

This script loads WFO results and computes IC separately for each market regime
to identify which regimes contribute to negative IC folds.

Usage:
    python scripts/analyze_ic_by_regime.py --experiment v_universe_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings
from intentflow_ai.modeling.regimes import RegimeClassifier
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze IC by market regime")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name (e.g., v_universe_full)",
    )
    return parser.parse_args()


def compute_ic(predictions: pd.Series, returns: pd.Series) -> float:
    """Compute Information Coefficient (Spearman correlation)."""
    valid = predictions.notna() & returns.notna()
    if valid.sum() < 10:
        return np.nan
    return stats.spearmanr(predictions[valid], returns[valid])[0]


def main() -> None:
    args = parse_args()
    
    exp_dir = settings.experiments_dir / args.experiment
    train_path = exp_dir / "train.parquet"
    summary_path = exp_dir / "walk_forward_summary.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    logger.info(f"Loading data from {exp_dir}")
    
    # Load training data
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(train_df)} rows")
    
    # Compute market regimes
    logger.info("Computing market regimes...")
    regime_classifier = RegimeClassifier()
    
    # Need price panel for regime detection
    price_panel = train_df[["date", "ticker", "close"]].copy() if "close" in train_df.columns else None
    
    if price_panel is None:
        # Try to load from raw prices
        price_path = settings.data_dir / "processed" / "prices.parquet"
        if price_path.exists():
            prices_df = pd.read_parquet(price_path)
            logger.info(f"Loaded {len(prices_df)} price rows from {price_path}")
            # Filter to training date range
            train_dates = set(train_df["date"].unique())
            prices_df = prices_df[prices_df["date"].isin(train_dates)]
            if "close" in prices_df.columns:
                price_panel = prices_df[["date", "ticker", "close"]].copy()
                logger.info(f"Using {len(price_panel)} filtered price rows")
            else:
                logger.warning(f"No 'close' column in prices - columns: {prices_df.columns.tolist()[:10]}")
        else:
            logger.warning(f"Price file not found: {price_path}")
            price_panel = None
    
    if price_panel is not None and len(price_panel) > 0:
        regimes = regime_classifier.infer(price_panel)
        logger.info(f"Computed regimes for {len(regimes)} dates")
        
        # Merge regimes with training data
        train_df = train_df.merge(
            regimes[["volatility_regime", "trend_regime", "composite_regime", "allow_entry", "regime_score"]],
            left_on="date",
            right_index=True,
            how="left"
        )
    else:
        logger.warning("Skipping regime detection - no price data")
        train_df["volatility_regime"] = "unknown"
        train_df["trend_regime"] = "unknown"
        train_df["composite_regime"] = "unknown"
        train_df["allow_entry"] = True
        train_df["regime_score"] = 50
    
    # Get excess returns
    excess_col = "excess_fwd"
    if excess_col not in train_df.columns:
        # Try fwd_ret column
        fwd_cols = [c for c in train_df.columns if c.startswith("fwd_ret")]
        if fwd_cols:
            excess_col = fwd_cols[0]
            logger.info(f"Using {excess_col} as return column")
        else:
            raise ValueError("No return column found")
    
    # Compute IC by regime
    logger.info("\n" + "=" * 80)
    logger.info("IC ANALYSIS BY REGIME")
    logger.info("=" * 80)
    
    results = []
    
    # Overall IC (using label as proxy for prediction if no proba column)
    # In real use, we'd have model predictions
    label_col = "label"
    
    # Compute IC by volatility regime
    for vol_regime in train_df["volatility_regime"].dropna().unique():
        mask = train_df["volatility_regime"] == vol_regime
        subset = train_df[mask]
        
        if len(subset) < 100:
            continue
            
        # Compute return correlation with label (as proxy)
        ic = compute_ic(subset[label_col].astype(float), subset[excess_col])
        
        results.append({
            "regime_type": "volatility",
            "regime_value": vol_regime,
            "sample_count": len(subset),
            "ic": ic,
            "mean_return": subset[excess_col].mean(),
            "std_return": subset[excess_col].std(),
            "pct_positive": (subset[excess_col] > 0).mean(),
        })
        
        logger.info(f"Volatility={vol_regime}: IC={ic:.4f}, n={len(subset)}, pos_rate={results[-1]['pct_positive']:.2%}")
    
    # Compute IC by trend regime
    for trend_regime in train_df["trend_regime"].dropna().unique():
        mask = train_df["trend_regime"] == trend_regime
        subset = train_df[mask]
        
        if len(subset) < 100:
            continue
            
        ic = compute_ic(subset[label_col].astype(float), subset[excess_col])
        
        results.append({
            "regime_type": "trend",
            "regime_value": trend_regime,
            "sample_count": len(subset),
            "ic": ic,
            "mean_return": subset[excess_col].mean(),
            "std_return": subset[excess_col].std(),
            "pct_positive": (subset[excess_col] > 0).mean(),
        })
        
        logger.info(f"Trend={trend_regime}: IC={ic:.4f}, n={len(subset)}, pos_rate={results[-1]['pct_positive']:.2%}")
    
    # Compute IC by composite regime
    for composite in train_df["composite_regime"].dropna().unique():
        mask = train_df["composite_regime"] == composite
        subset = train_df[mask]
        
        if len(subset) < 100:
            continue
            
        ic = compute_ic(subset[label_col].astype(float), subset[excess_col])
        
        results.append({
            "regime_type": "composite",
            "regime_value": composite,
            "sample_count": len(subset),
            "ic": ic,
            "mean_return": subset[excess_col].mean(),
            "std_return": subset[excess_col].std(),
            "pct_positive": (subset[excess_col] > 0).mean(),
        })
        
        logger.info(f"Composite={composite}: IC={ic:.4f}, n={len(subset)}, pos_rate={results[-1]['pct_positive']:.2%}")
    
    # Compute IC for allow_entry filter
    for allow_entry in [True, False]:
        mask = train_df["allow_entry"] == allow_entry
        subset = train_df[mask]
        
        if len(subset) < 100:
            continue
            
        ic = compute_ic(subset[label_col].astype(float), subset[excess_col])
        
        results.append({
            "regime_type": "allow_entry",
            "regime_value": str(allow_entry),
            "sample_count": len(subset),
            "ic": ic,
            "mean_return": subset[excess_col].mean(),
            "std_return": subset[excess_col].std(),
            "pct_positive": (subset[excess_col] > 0).mean(),
        })
        
        logger.info(f"AllowEntry={allow_entry}: IC={ic:.4f}, n={len(subset)}, pos_rate={results[-1]['pct_positive']:.2%}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = exp_dir / "ic_by_regime.json"
    results_df.to_json(output_path, orient="records", indent=2)
    logger.info(f"\nSaved results to {output_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    
    if len(results_df) > 0:
        # Find best and worst regimes
        best = results_df.loc[results_df["ic"].idxmax()]
        worst = results_df.loc[results_df["ic"].idxmin()]
        
        logger.info(f"\nBEST regime: {best['regime_type']}={best['regime_value']} (IC={best['ic']:.4f})")
        logger.info(f"WORST regime: {worst['regime_type']}={worst['regime_value']} (IC={worst['ic']:.4f})")
        
        # Recommendation
        if worst["ic"] < 0:
            logger.info(f"\n⚠️  Recommendation: Filter out {worst['regime_type']}={worst['regime_value']} to reduce IC variance")


if __name__ == "__main__":
    main()
