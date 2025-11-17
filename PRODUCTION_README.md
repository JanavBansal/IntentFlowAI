# Production Alpha Model - NIFTY200

## Overview

This is a **production-ready, fully interpretable, and systematically stress-tested** alpha model for trading the NIFTY200 universe. It implements best practices for institutional-grade quantitative trading systems.

## Key Features

### 🔒 Anti-Overfitting & Data Leakage Prevention
- **Time-purged splits** with embargoed periods to prevent lookahead bias
- **Walk-forward validation** with rolling time windows
- **Null-label testing**: Ensures performance isn't due to data leakage by shuffling labels and verifying collapse
- **Forward alignment verification**: Validates that labels only use future data

### 🧠 Model Robustness & Stability
- **Stability-optimized hyperparameters**: Prefers consistent out-of-sample performance over peak in-sample fit
- **Cross-validation variance minimization**: Low variance across time folds
- **Baseline benchmarking**: Compares against linear models and trivial baselines
- **Conservative regularization**: Enforces L1/L2 penalties to prevent overfitting

### 🌍 Market Regime Awareness
- **Multi-dimensional regime detection**:
  - Volatility regimes (low/medium/high/extreme)
  - Trend regimes (strong uptrend → strong downtrend)
  - Market breadth (% of stocks above MA)
  - Drawdown monitoring
- **Regime-based trade filtering**: Blocks trades in unfavorable conditions (high vol, downtrends)
- **Regime score (0-100)**: Quantifies market favorability
- **Regime segmentation**: Separate models trained per regime

### 🔍 Maximum Interpretability
- **SHAP explanations** for every prediction
- **Signal cards** with complete rationale:
  - Top contributing features with values and SHAP contributions
  - Market regime context
  - Confidence level and risk warnings
  - Historical performance of similar signals
- **Feature importance tracking**: Monitors stability across retrains
- **Audit trail**: Every signal includes full feature values for review

### 🧪 Feature Engineering Excellence
- **Orthogonality testing**: Removes redundant/correlated features
  - Spearman correlation analysis
  - Hierarchical clustering
  - Variance Inflation Factor (VIF) for multicollinearity
- **Incremental IC testing**: Only adds features that improve out-of-sample IC
- **Feature drift detection**: Monitors distribution shifts using KS tests and PSI

### 💪 Comprehensive Stress Testing
Tests every scenario before deployment:
- **Transaction cost sensitivity**: Slippage from 5bps to 100bps
- **Volatility shocks**: 1.5x to 5x amplification
- **Market crashes**: -10% to -40% drawdowns
- **Parameter robustness**: Perturbations to top_k, holding period, etc.
- **Monte Carlo simulation**: 1000+ bootstrap runs with block resampling

### 📊 Real-Time Monitoring & Drift Detection
- **Feature drift alerts**: KS statistic & PSI thresholds
- **Performance degradation tracking**: IC, Sharpe, hit rate monitoring
- **Prediction drift**: Detects model calibration issues
- **Health score (0-100)**: Overall system health
- **Automated retrain triggers**: When drift exceeds thresholds

### 🎯 Meta-Labeling & Risk Filters
- **Secondary meta-model**: Predicts which primary signals to take
- **Regime-aware filtering**: Only trades in favorable conditions
- **Position limits**: Max concurrent positions
- **Cooldown periods**: Prevents over-trading same ticker
- **Drawdown-based shutoff**: Stops trading in large drawdowns

## Production Pipeline

Run the complete production pipeline:

```bash
python scripts/run_production_pipeline.py --experiment production_v1
```

### Pipeline Stages

1. **Data Loading & Feature Engineering**
   - Load NIFTY200 price panel
   - Engineer 50+ technical & fundamental features
   - Generate excess return labels

2. **Feature Orthogonality Analysis**
   - Test correlation, VIF, clustering
   - Drop redundant features
   - Generate orthogonality report

3. **Time-Purged Splits**
   - Train/valid/test with embargoed periods
   - No lookahead bias

4. **Market Regime Detection**
   - Classify volatility, trend, breadth
   - Generate regime scores
   - Save regime data

5. **Stability-Optimized Training**
   - Grid search with stability scoring
   - Optimize for low CV variance
   - Test parameter robustness

6. **Baseline Comparison**
   - Compare to linear model
   - Ensure non-trivial improvement

7. **Null Label Test (Leakage Detection)**
   - Shuffle labels by date blocks
   - Verify performance collapses
   - Confirm no data leakage

8. **SHAP Explanations**
   - Generate SHAP values for all predictions
   - Store for signal cards

9. **Regime-Filtered Signals**
   - Apply regime filters
   - Block unfavorable trades

10. **Signal Cards Generation**
    - Full interpretability cards
    - Markdown + JSON outputs

11. **Comprehensive Stress Testing**
    - Test all scenarios
    - Generate stress test report

12. **Backtest**
    - Simulate live trading
    - Calculate Sharpe, CAGR, drawdown

13. **Model Evaluation**
    - ROC AUC, IC, Rank IC
    - Precision@K, hit rates

14. **Drift Detection**
    - Compare test vs train distributions
    - Generate drift alerts

15. **Production Readiness Assessment**
    - Automated go/no-go decision
    - Performance, stress, drift checks

## Output Files

All results saved to `experiments/{experiment_name}/`:

```
├── PRODUCTION_SUMMARY.md          # Executive summary with go/no-go verdict
├── feature_importance.csv         # Feature importance rankings
├── orthogonality_report.md        # Feature correlation analysis
├── stability_report.md            # Model stability optimization results
├── baseline_comparison.json       # Comparison to linear baseline
├── regime_summary.json            # Regime distribution stats
├── regime_data.csv                # Daily regime classifications
├── null_test/                     # Null label test results
│   ├── null_label_preds.csv
│   └── null_label_summary.json
├── top_signals.csv                # Top 100 trading signals
├── signal_cards/                  # Full interpretability cards
│   ├── signal_cards.json
│   └── *.md                       # Individual signal cards
├── stress_tests/                  # Comprehensive stress testing
│   ├── stress_test_summary.json
│   ├── stress_test_results.csv
│   ├── monte_carlo_results.json
│   └── stress_test_report.md
├── bt_equity.csv                  # Backtest equity curve
├── bt_trades.csv                  # All trades
├── bt_summary.json                # Backtest metrics
├── metrics.json                   # Model evaluation metrics
├── drift_monitoring/              # Drift detection results
│   ├── drift_alerts.json
│   ├── drift_report.json
│   └── drift_summary.md
└── universe_snapshot.csv          # Tickers used in this run
```

## Production Readiness Checklist

The pipeline automatically checks:

- ✅ **ROC AUC > 0.55** (better than random)
- ✅ **Sharpe Ratio > 0.5** (positive risk-adjusted returns)
- ✅ **Max Drawdown < -25%** (acceptable risk)
- ✅ **Stress test pass rate > 50%** (robust to adverse conditions)
- ✅ **Health score > 60** (no critical drift)

## Live Trading Dashboard

Run the Streamlit dashboard for real-time monitoring:

```bash
streamlit run dashboard/app.py
```

Features:
- Real-time signal display with SHAP explanations
- Rolling IC chart
- Exposure metrics by sector
- Feature drift detection
- Backtest performance visualization
- Auto-refresh capability

## Configuration

Key settings in `config/experiments/*.yaml`:

```yaml
trainer:
  model: lightgbm
  params:
    learning_rate: 0.03        # Conservative
    n_estimators: 600
    max_depth: -1
    reg_lambda: 1.0            # L2 regularization
    feature_fraction: 0.7      # Feature subsampling

risk_filters:
  vol_high: 0.04               # Block trades if vol > 4%
  allow_high_vol: false
  allow_downtrend: false
  max_positions: 12            # Position limits
  cooldown_days: 2

meta_labeling:
  enabled: true                # Use meta-labeling
  success_threshold: 0.015     # 1.5% return threshold
  min_signal_proba: 0.55       # Filter weak signals
```

## Key Modules

### `intentflow_ai.modeling.regimes`
Multi-dimensional market regime detection with volatility, trend, and breadth signals.

### `intentflow_ai.modeling.stability`
Stability-optimized hyperparameter tuning that prefers robust parameters over peak performance.

### `intentflow_ai.modeling.signal_cards`
Production-grade signal cards with complete interpretability, SHAP explanations, and regime context.

### `intentflow_ai.features.orthogonality`
Feature orthogonality testing with correlation, VIF, and clustering analysis.

### `intentflow_ai.sanity.stress_tests`
Comprehensive stress testing framework for transaction costs, volatility shocks, crashes, and Monte Carlo.

### `intentflow_ai.monitoring.drift_detection`
Real-time drift detection with automated alerts and retrain triggers.

## Best Practices

### Training
1. Always use time-purged splits with embargo periods
2. Run null-label tests to verify no leakage
3. Optimize for stability, not peak performance
4. Compare to linear baselines
5. Test feature orthogonality

### Signal Generation
1. Generate SHAP explanations for every signal
2. Apply regime filters
3. Create full signal cards with rationale
4. Include risk warnings
5. Track confidence levels

### Monitoring
1. Run drift detection daily
2. Monitor rolling IC
3. Track regime distributions
4. Alert on health score drops
5. Retrain when drift exceeds thresholds

### Risk Management
1. Block trades in high volatility
2. Avoid strong downtrends
3. Limit max positions
4. Use cooldown periods
5. Stop trading on large drawdowns

## Target Metrics

Production model should achieve:
- **ROC AUC**: > 0.55 (test set)
- **Rank IC**: > 0.03 (test set)
- **Sharpe Ratio**: > 1.0 (backtest)
- **Win Rate**: > 50%
- **Max Drawdown**: < -25%
- **Stress Test Pass Rate**: > 70%
- **Health Score**: > 70

## Automated Alerts

The system triggers alerts for:
- 🚨 **Critical drift**: Feature or prediction distributions shift significantly
- 📉 **Performance degradation**: IC, Sharpe, or hit rate drops below thresholds
- ⚠️ **Regime shift**: Market enters unfavorable regime
- 🔄 **Retrain recommended**: Automated trigger when drift is severe
- 🛑 **Stop trading**: Critical health score drop

## Contact & Support

For questions or issues, refer to the main README or consult the model documentation.

---

**Remember**: This is a systematic trading system. Every trade decision should be backed by:
1. High model probability (>60%)
2. Favorable regime (score >30)
3. Clear SHAP rationale
4. Acceptable confidence level
5. No critical warnings

**Never override the system without documented justification.**

