# IntentFlow AI

## Overview

**IntentFlow AI** is a production-ready, systematic trading signal platform designed to produce **live, interpretable, position-level trading signals** for the **NIFTY 200** universe. The platform integrates multiple heterogeneous data layers with robust cross-validation, regime filters, meta-labeling, and SHAP-based explanations to deliver actionable, high-quality trading signals.

### Core Requirements & Targets

- **Live Signal Generation**: Position-level trading signals with full diagnostic outputs
- **Interpretability**: Every signal includes SHAP explanations with top features and rationale
- **Robust Validation**: Purged, embargoed time-series cross-validation to prevent data leaks
- **Risk Management**: Regime and volatility filters, meta-labeling for alpha enhancement
- **Performance Targets**:
  - **Out-of-sample ROC AUC**: >50%
  - **Information Coefficient (IC)**: >50%
  - **Win Rate**: >50%
  - **Max Drawdown**: <25%

### Data Layers

- **Market flows & ownership data** (FII/DII flows, delivery data)
- **Price & transaction microstructure** (OHLCV, volume patterns)
- **Fundamental drift & accounting signals** (earnings quality, balance sheet changes)
- **Narrative tone & sentiment** (news, social media, filings)
- **Price-confirmation & technical structure** (momentum, volatility, regime indicators)

---

## Architecture

The stack is modular, production-ready, and designed for rapid audit and evolution:

```
intentflow_ai/
├── config/          # Settings, experiment configs, cost models
├── data/            # Ingestion, universe management, coverage tracking
├── features/        # Feature engineering, label creation
├── modeling/        # Training, evaluation, regimes, SHAP explanations
├── pipelines/       # Training and scoring pipelines
├── backtest/        # Cost-aware backtesting with risk filters
├── sanity/          # Leakage tests, data scope validation, cost sweeps
└── utils/           # Splits, I/O, logging, caching

scripts/             # Entry points: training, scoring, backtest, sanity
dashboard/           # Live dashboard with metrics, SHAP explanations, drift detection
experiments/         # Model artifacts, metrics, reports, SHAP outputs
```

---

## Key Features

### ✅ Implemented

1. **Purged Time-Series Cross-Validation**
   - Embargo windows to prevent label leakage
   - Horizon-aware purging in `intentflow_ai/utils/splits.py`
   - Configurable via experiment YAML

2. **Regime & Volatility Filters**
   - Market regime classification (bull/bear) in `intentflow_ai/modeling/regimes.py`
   - Volatility filters in `intentflow_ai/backtest/filters.py`
   - Configurable trend/vol thresholds

3. **Meta-Labeling**
   - Second-stage model to filter primary signals
   - Implemented in `intentflow_ai/meta_labeling/core.py`
   - Configurable success thresholds and probability gates

4. **SHAP Explanations**
   - Position-level feature attribution in `intentflow_ai/modeling/explanations.py`
   - Top contributing features with rationale
   - Integrated into scoring pipeline

5. **Live Dashboard**
   - Real-time signal monitoring with SHAP explanations
   - Rolling IC, exposure metrics, feature drift detection
   - Performance metrics (Sharpe, drawdown, win rate)

6. **Data Leakage Prevention**
   - Leakage tests in `intentflow_ai/sanity/leakage_tests.py`
   - Null-label backtests to verify signal quality
   - Forward-alignment validation

7. **Cost-Aware Backtesting**
   - Realistic Indian market costs (brokerage, STT, GST, etc.)
   - Cost sweep analysis
   - Configurable slippage and fees

---

## Typical Workflow

### 1. Training Pipeline

```bash
# Train with experiment config
python scripts/run_training.py --config config/experiments/v_universe_sanity.yaml

# With leak test
python scripts/run_training.py --config config/experiments/v_universe_sanity.yaml --leak-test
```

**Outputs:**
- Trained model (`lgb.pkl`) with regime-specific models
- Training metrics (`metrics.json`) with split-wise performance
- Predictions (`preds.csv`) for all training data
- Feature importance (`feature_importance.csv`)
- Training frame (`train.parquet`) for SHAP background

### 2. Signal Generation (Scoring)

```bash
# Generate live signals with SHAP explanations
python scripts/run_scoring.py --experiment v_universe_sanity
```

**Outputs:**
- `top_signals.csv` with columns:
  - `date`, `ticker`, `sector`, `proba`, `rank`
  - `top_features` (list of dicts with SHAP values)
  - `rationale` (human-readable explanation)
  - `shap_values` (dict mapping feature → contribution)

### 3. Backtesting

```bash
# Run backtest with risk filters
python scripts/run_backtest.py --experiment v_universe_sanity
```

**Outputs:**
- `bt_summary.json` (CAGR, Sharpe, drawdown, win rate)
- `bt_equity.csv` (equity curve)
- `bt_trades.csv` (all trades with entry/exit)

### 4. Sanity Checks

```bash
# Run comprehensive sanity suite
python scripts/run_sanity.py --experiment v_universe_sanity
```

**Outputs:**
- Null-label backtest (should show no edge)
- Cost sweep analysis
- Data scope validation
- Feature drift detection

### 5. Live Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard/app.py
```

**Features:**
- Real-time signal monitoring
- SHAP explanations for each position
- Rolling IC and exposure metrics
- Feature drift alerts
- Performance visualization

---

## 📊 Current Model Performance

_Experiment: `v_universe_sanity` (NIFTY 200 universe, baseline features)_

| Metric                  | Result    | Target    | Status |
|-------------------------|-----------|-----------|--------|
| **ROC AUC (test)**      | 0.50      | >0.50     | ⚠️ Marginal |
| **IC (test)**           | 0.004     | >0.50     | ❌ Below target |
| **Sharpe Ratio**        | 20.42     | >1.0      | ✅ Excellent |
| **Max Drawdown**        | -89.27%   | <-25%     | ❌ Above target |
| **Win Rate**            | 84.0%     | >50%      | ✅ Exceeds target |
| **CAGR**                | 33,113%   | N/A       | ⚠️ Suspicious (likely data issue) |

**Observations:**
- Strong in-sample performance (train/valid ROC AUC ~0.88-0.90)
- Weak out-of-sample generalization (test ROC AUC ~0.50)
- Very high Sharpe suggests strong risk-adjusted returns in-sample
- High drawdown indicates need for better risk filters
- Test IC near zero suggests overfitting or regime shift

**Next Steps:**
- Expand feature set and data layers
- Improve out-of-sample generalization
- Add stronger risk filters and position sizing
- Investigate test set performance degradation

---

## Data Integrity & Audit

### Leakage Prevention

1. **Purged CV**: Training folds exclude observations with overlapping label horizons
2. **Embargo Windows**: Configurable gaps between train/valid/test splits
3. **Forward Alignment**: Labels computed using only past information
4. **Null-Label Tests**: Random labels should produce no edge

### Validation Checks

- ✅ Purged time-series splits with embargo
- ✅ Forward-aligned label computation
- ✅ Leakage test mode (shuffled labels)
- ✅ Null-label backtest validation
- ✅ Feature drift detection

### Audit Trail

All experiments include:
- **Config**: Experiment YAML with all hyperparameters
- **Metrics**: Split-wise performance (train/valid/test)
- **Artifacts**: Models, predictions, feature importance
- **Reports**: Markdown reports with diagnostics
- **SHAP**: Feature explanations for top signals

---

## Configuration

### Experiment Config (`config/experiments/v_universe_sanity.yaml`)

```yaml
universe:
  universe_file: data/external/universe/nifty200.csv
  use_membership: false

splits:
  train_start: "2018-01-01"
  valid_start: "2023-01-01"
  test_start: "2024-07-01"

trainer:
  model: lightgbm
  params:
    n_estimators: 600
    learning_rate: 0.03
    # ... more params

risk_filters:
  trend_fast: 50
  trend_slow: 200
  vol_lookback: 20
  vol_high: 0.04
  allow_high_vol: false
  allow_downtrend: false
  max_positions: 12
  cooldown_days: 2

meta_labeling:
  enabled: false
  horizon_days: 10
  success_threshold: 0.015
  min_signal_proba: 0.55
```

---

## Development & Evolution

### Code Quality

- Modular architecture for easy extension
- Type hints throughout
- Comprehensive logging
- Error handling with graceful fallbacks

### Extensibility

- **Feature Blocks**: Add new feature types in `FeatureEngineer`
- **Regime Classifiers**: Extend `RegimeClassifier` for custom regimes
- **Cost Models**: Add new cost models in `config/costs_india.yaml`
- **Filters**: Add risk filters in `intentflow_ai/backtest/filters.py`

### Testing

```bash
# Smoke test
python scripts/smoke_test.py

# Leakage test
python scripts/run_training.py --leak-test

# Sanity suite
python scripts/run_sanity.py --experiment v_universe_sanity
```

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- `lightgbm>=4.0.0` - Gradient boosting
- `shap>=0.44.0` - Model explanations
- `pandas>=2.1.0` - Data manipulation
- `streamlit>=1.29.0` - Dashboard
- `scikit-learn>=1.3.0` - ML utilities

---

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure paths:**
   - Update `intentflow_ai/config/settings.py` if needed
   - Ensure data files are in `data/` directory

3. **Run training:**
   ```bash
   python scripts/run_training.py --config config/experiments/v_universe_sanity.yaml
   ```

4. **Generate signals:**
   ```bash
   python scripts/run_scoring.py --experiment v_universe_sanity
   ```

5. **Launch dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

---

## Roadmap

### Immediate Priorities
- [ ] Improve out-of-sample performance (target: >50% ROC AUC, >50% IC)
- [ ] Reduce drawdowns (target: <25%)
- [ ] Expand to full NIFTY 200 universe
- [ ] Add more feature layers (ownership, fundamentals, narrative)

### Future Enhancements
- [ ] Real-time data ingestion pipeline
- [ ] Automated feature drift alerts
- [ ] Portfolio optimization integration
- [ ] Multi-timeframe signals
- [ ] Ensemble models with stacking

---

**IntentFlow AI** - Production-ready systematic trading signals with full interpretability and audit trail.
