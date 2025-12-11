#!/usr/bin/env python
"""
IntentFlow v2 Demo Script

Demonstrates the complete conviction engine pipeline:
1. Load data and compute features
2. Detect market regime (HMM)
3. Train model with calibration
4. Generate monthly conviction rankings

Usage:
    python scripts/run_conviction_engine.py
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.labels import make_excess_label
from intentflow_ai.modeling.ensemble import MultiAlgoEnsemble
from intentflow_ai.modeling.regime_detector import RegimeDetector, compute_regime_features
from intentflow_ai.modeling.calibration import ProbabilityCalibrator
from intentflow_ai.features.india_alpha import compute_india_alpha_features
from intentflow_ai.data.india_market_data import IndiaVIXProvider, FIIDIIProvider
from intentflow_ai.ranking.conviction_ranker import ConvictionRanker, RankingConfig

logger = get_logger(__name__)


def run_conviction_engine(
    output_dir: str = "rankings",
    horizon_days: int = 30,
    calibration_method: str = "isotonic"
):
    """
    Run the complete IntentFlow v2 conviction engine.
    
    Steps:
    1. Load and prepare data
    2. Compute regime features and detect current regime
    3. Train ensemble model with probability calibration
    4. Generate monthly conviction rankings
    5. Export ranking report
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("Step 1: Loading data...")
    
    prices = load_price_parquet("data/processed/prices.parquet")
    logger.info(f"Loaded {len(prices)} price rows for {prices['ticker'].nunique()} tickers")
    
    # Load India VIX
    vix_provider = IndiaVIXProvider()
    vix_data = vix_provider.fetch(start_date="2020-01-01")
    logger.info(f"Loaded {len(vix_data)} days of India VIX data")
    
    # Load FII/DII data (synthetic for now)
    fii_provider = FIIDIIProvider()
    fii_data = fii_provider.fetch(start_date="2020-01-01")
    logger.info(f"Loaded {len(fii_data)} days of FII/DII data")
    
    # =========================================================================
    # Step 2: Regime Detection
    # =========================================================================
    logger.info("Step 2: Detecting market regime...")
    
    regime_features = compute_regime_features(prices, vix_data, fii_data)
    
    detector = RegimeDetector()
    detector.fit(regime_features)
    
    current_regime = detector.predict_regime(regime_features.tail(30))
    should_trade = detector.should_trade()
    
    logger.info(f"Current Regime: {current_regime}")
    logger.info(f"Should Trade: {should_trade}")
    
    if not should_trade:
        logger.warning("Regime is not favorable. Rankings will be generated but flagged.")
    
    # =========================================================================
    # Step 3: Feature Engineering + India Alpha
    # =========================================================================
    logger.info("Step 3: Engineering features...")
    
    # Standard features
    engineer = FeatureEngineer()
    base_features = engineer.build(prices)
    
    # India alpha features
    india_features = compute_india_alpha_features(prices, fii_data)
    
    # Merge features
    features = base_features.copy()
    if not india_features.empty:
        # Add India features by index alignment
        india_cols = [c for c in india_features.columns if c not in ["date", "ticker"]]
        for col in india_cols:
            if col in india_features.columns:
                features[f"india__{col}"] = india_features[col].values[:len(features)]
    
    logger.info(f"Total features: {len(features.columns)}")
    
    # =========================================================================
    # Step 4: Labels and Train/Calib Split
    # =========================================================================
    logger.info("Step 4: Creating labels and splitting data...")
    
    labeled = make_excess_label(prices, horizon_days=horizon_days, thresh=0.03)
    
    # Align features with labels
    common_idx = features.index.intersection(labeled.index)
    X = features.loc[common_idx]
    y = labeled.loc[common_idx, "label"]
    
    # Split: 70% train, 15% calibration, 15% test
    n = len(X)
    n_train = int(n * 0.70)
    n_calib = int(n * 0.15)
    
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_calib, y_calib = X.iloc[n_train:n_train+n_calib], y.iloc[n_train:n_train+n_calib]
    X_test, y_test = X.iloc[n_train+n_calib:], y.iloc[n_train+n_calib:]
    
    logger.info(f"Train: {len(X_train)}, Calib: {len(X_calib)}, Test: {len(X_test)}")
    
    # =========================================================================
    # Step 5: Train Model
    # =========================================================================
    logger.info("Step 5: Training ensemble model...")
    
    model = MultiAlgoEnsemble()
    model.train(X_train, y_train)
    
    # =========================================================================
    # Step 6: Calibrate Probabilities
    # =========================================================================
    logger.info("Step 6: Calibrating probabilities...")
    
    calibrator = ProbabilityCalibrator(method=calibration_method)
    calibrator.fit(model, X_calib, y_calib)
    
    metrics = calibrator.get_metrics()
    if metrics:
        logger.info(f"Calibration Brier Score: {metrics.brier_score:.4f}")
        
        # Save reliability diagram
        calibrator.plot_reliability_diagram(
            save_path=str(output_path / "reliability_diagram.png")
        )
    
    # =========================================================================
    # Step 7: Generate Monthly Rankings
    # =========================================================================
    logger.info("Step 7: Generating monthly conviction rankings...")
    
    # Use test set features for ranking
    ranker_config = RankingConfig(
        min_probability_to_rank=0.40,
        top_n=50,
        shap_top_k=3,
        require_risk_on_regime=False  # Generate even if not risk-on
    )
    
    ranker = ConvictionRanker(
        base_model=model,
        calibrator=calibrator,
        regime_detector=detector,
        config=ranker_config
    )
    
    # Get latest features for ranking
    latest_features = X_test.copy()
    latest_features["ticker"] = labeled.loc[X_test.index, "ticker"].values
    
    rankings = ranker.generate_monthly_ranking(
        features_df=latest_features,
        prices_df=prices,
        regime_features=regime_features.tail(30)
    )
    
    # =========================================================================
    # Step 8: Export Results
    # =========================================================================
    logger.info("Step 8: Exporting results...")
    
    # Export CSV
    ranking_date = datetime.now().strftime("%Y-%m")
    csv_path = output_path / f"conviction_ranking_{ranking_date}.csv"
    ranker.export_ranking(rankings, str(csv_path))
    
    # Export markdown report
    report_path = output_path / f"conviction_report_{ranking_date}.md"
    report = ranker.generate_ranking_report(rankings, str(report_path))
    
    # Summary
    logger.info("=" * 60)
    logger.info("CONVICTION ENGINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Regime: {current_regime.value if hasattr(current_regime, 'value') else current_regime}")
    logger.info(f"Should Trade: {should_trade}")
    logger.info(f"Tickers Ranked: {len(rankings)}")
    logger.info(f"Calibration Brier: {metrics.brier_score:.4f}" if metrics else "N/A")
    logger.info(f"Output: {output_path}")
    
    # Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 CONVICTION PICKS")
    print("=" * 60)
    if not rankings.empty and "rank" in rankings.columns:
        print(rankings.head(10).to_string(index=False))
    else:
        print("No rankings generated")
    
    return rankings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntentFlow v2 Conviction Engine")
    parser.add_argument("--output", default="rankings", help="Output directory")
    parser.add_argument("--horizon", type=int, default=30, help="Forward horizon days")
    parser.add_argument("--calib", choices=["sigmoid", "isotonic"], default="isotonic")
    
    args = parser.parse_args()
    
    run_conviction_engine(
        output_dir=args.output,
        horizon_days=args.horizon,
        calibration_method=args.calib
    )
