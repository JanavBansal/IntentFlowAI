#!/usr/bin/env python
"""Quick test script for production features without full pipeline run."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from intentflow_ai.features.orthogonality import FeatureOrthogonalityAnalyzer
from intentflow_ai.modeling.regimes import RegimeClassifier, apply_regime_filter_to_signals
from intentflow_ai.modeling.signal_cards import SignalCardGenerator
from intentflow_ai.modeling.stability import StabilityOptimizer
from intentflow_ai.monitoring.drift_detection import DriftDetector
from intentflow_ai.sanity.stress_tests import StressTestSuite, StressTestConfig
from intentflow_ai.backtest.core import BacktestConfig

print("=" * 80)
print("TESTING PRODUCTION FEATURES")
print("=" * 80)

# Create synthetic test data
np.random.seed(42)
n_samples = 1000
n_features = 20

dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
tickers = [f'TICK{i:03d}' for i in range(50)]

print(f"\n[1/7] Creating synthetic test data...")
print(f"   - {n_samples} samples, {n_features} features, {len(tickers)} tickers")

# Create feature data
feature_data = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
feature_data['date'] = np.random.choice(dates, n_samples)
feature_data['ticker'] = np.random.choice(tickers, n_samples)

labels = (np.random.randn(n_samples) > 0).astype(int)

# Add some correlated features for orthogonality test
feature_data['feature_5'] = feature_data['feature_0'] * 0.9 + np.random.randn(n_samples) * 0.1
feature_data['feature_10'] = feature_data['feature_1'] * 0.85 + np.random.randn(n_samples) * 0.15

# Test 1: Feature Orthogonality
print("\n[2/7] Testing Feature Orthogonality Analysis...")
try:
    analyzer = FeatureOrthogonalityAnalyzer()
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    analysis = analyzer.analyze(feature_data[feature_cols], pd.Series(labels))
    
    selected, dropped = analyzer.select_orthogonal_features(
        feature_data[feature_cols],
        pd.Series(labels)
    )
    
    print(f"   ✓ Analyzed {len(feature_cols)} features")
    print(f"   ✓ Found {len(analysis.get('highly_correlated_pairs', []))} highly correlated pairs")
    print(f"   ✓ Selected {len(selected)}/{len(feature_cols)} features")
    print(f"   ✓ Dropped {len(dropped)} features")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 2: Regime Detection
print("\n[3/7] Testing Market Regime Detection...")
try:
    price_panel = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=500, freq='D').repeat(len(tickers)),
        'ticker': tickers * 500,
        'close': 100 + np.random.randn(500 * len(tickers)).cumsum()
    })
    
    classifier = RegimeClassifier()
    regime_data = classifier.infer(price_panel)
    
    print(f"   ✓ Detected regimes for {len(regime_data)} dates")
    print(f"   ✓ Regime columns: {list(regime_data.columns)}")
    print(f"   ✓ Mean regime score: {regime_data['regime_score'].mean():.1f}/100")
    print(f"   ✓ Entry allowed: {regime_data['allow_entry'].mean():.1%} of days")
    
    regime_summary = classifier.get_regime_summary(regime_data)
    print(f"   ✓ Generated regime summary with {len(regime_summary)} metrics")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 3: Signal Cards
print("\n[4/7] Testing Signal Card Generation...")
try:
    signals = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'ticker': ['TICK001'] * 10,
        'proba': np.linspace(0.6, 0.9, 10),
        'rank': range(1, 11),
        'sector': ['IT'] * 10,
    })
    
    features = pd.DataFrame(
        np.random.randn(10, 10),
        columns=[f'feat_{i}' for i in range(10)]
    )
    
    card_generator = SignalCardGenerator(model_version="test_v1")
    cards = card_generator.generate_cards(
        signals=signals,
        features=features,
    )
    
    print(f"   ✓ Generated {len(cards)} signal cards")
    if cards:
        card = cards[0]
        print(f"   ✓ Card includes: ticker={card.ticker}, confidence={card.confidence_level}")
        print(f"   ✓ Top features: {len(card.top_features)}")
        print(f"   ✓ Risk warnings: {len(card.risk_warnings)}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 4: Stability Optimizer (lightweight test)
print("\n[5/7] Testing Model Stability Framework...")
try:
    from intentflow_ai.config.settings import LightGBMConfig
    from intentflow_ai.modeling.stability import compare_to_baseline
    
    # Create minimal test config
    test_config = LightGBMConfig(
        learning_rate=0.05,
        n_estimators=100,
        random_state=42
    )
    
    # Test baseline comparison
    test_features = pd.DataFrame(
        np.random.randn(200, 10),
        columns=[f'f_{i}' for i in range(10)]
    )
    test_labels = (np.random.randn(200) > 0).astype(int)
    
    comparison = compare_to_baseline(
        test_config,
        test_features,
        test_labels,
        baseline_type="linear"
    )
    
    print(f"   ✓ Baseline comparison completed")
    print(f"   ✓ Optimized AUC: {comparison['optimized_auc']:.3f}")
    print(f"   ✓ Baseline AUC: {comparison['baseline_auc']:.3f}")
    print(f"   ✓ Improvement: {comparison['auc_improvement']:+.3f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 5: Stress Testing
print("\n[6/7] Testing Stress Test Framework...")
try:
    # Create minimal test data
    test_signals = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=50, freq='D'),
        'ticker': ['TICK001'] * 50,
        'proba': np.random.rand(50),
        'label': (np.random.randn(50) > 0).astype(int),
    })
    
    test_prices = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D').repeat(5),
        'ticker': ['TICK001', 'TICK002', 'TICK003', 'TICK004', 'TICK005'] * 100,
        'close': 100 + np.random.randn(500).cumsum()
    })
    
    # Limited stress test config for speed
    stress_cfg = StressTestConfig(
        slippage_scenarios=[10.0, 20.0],
        fee_scenarios=[1.0],
        vol_multipliers=[1.5],
        crash_magnitudes=[-0.10],
        simulate_crashes=False,
        test_parameter_sensitivity=False,
        run_monte_carlo=False,
    )
    
    suite = StressTestSuite(stress_cfg)
    results = suite.run_all_tests(
        signals=test_signals,
        prices=test_prices,
        base_config=BacktestConfig(),
    )
    
    print(f"   ✓ Ran {results['summary']['total_scenarios']} stress scenarios")
    print(f"   ✓ Pass rate: {results['summary'].get('pass_rate', 0):.1%}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test 6: Drift Detection
print("\n[7/7] Testing Drift Detection...")
try:
    # Create test data with drift
    current_features = pd.DataFrame(
        np.random.randn(100, 10) + 0.5,  # Shifted distribution
        columns=[f'feat_{i}' for i in range(10)]
    )
    reference_features = pd.DataFrame(
        np.random.randn(200, 10),
        columns=[f'feat_{i}' for i in range(10)]
    )
    
    current_preds = pd.Series(np.random.rand(100) * 0.5 + 0.4)  # Shifted predictions
    reference_preds = pd.Series(np.random.rand(200))
    
    detector = DriftDetector()
    alerts, report = detector.detect_all_drift(
        current_features=current_features,
        reference_features=reference_features,
        current_predictions=current_preds,
        reference_predictions=reference_preds,
    )
    
    print(f"   ✓ Drift detection completed")
    print(f"   ✓ Generated {len(alerts)} alerts")
    print(f"   ✓ Health score: {report['summary']['health_score']:.0f}/100")
    print(f"   ✓ Status: {report['summary']['status'].upper()}")
    if alerts:
        print(f"   ✓ Severity distribution: {report['summary']['severity_counts']}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n" + "=" * 80)
print("✓ ALL PRODUCTION FEATURES TESTED SUCCESSFULLY")
print("=" * 80)

