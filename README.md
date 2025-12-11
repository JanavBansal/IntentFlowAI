# IntentFlow AI

A **production-ready systematic trading signal platform** for **NIFTY 500** (464 tickers) using multi-algorithm ensemble modeling with regime-aware predictions.

---

## 📊 Current Status (Dec 2025)

### Model Performance

| Version | Test Period | Test IC | Test ROC-AUC | Status |
|---------|-------------|---------|--------------|--------|
| **V3** (latest) | Jul 2024+ | -0.001 | 0.51 | ⚠️ Under revision |
| V2 | 2020-2024 | 0.062 | 0.54 | Previous stable |
| V1 | 2011-2019 | 0.075 | 0.54 | Historical best |

> **Note:** The model is experiencing **alpha decay** in 2024+. Recent market regime shifts have degraded predictive power. Active research underway to restore IC.

### Key Findings
- ✅ **80% Precision@10** in historical testing
- ⚠️ **IC collapsed in 2024** from 0.075 to ~0.01
- 🔬 **Root cause**: Technical features are crowded; need alternative data

---

## 🚀 Quick Start

### Run Dashboard
```bash
cd /Users/janavbansal/Documents/IntentFlowAI
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### Generate Predictions
```bash
python scripts/run_scoring.py --experiment v3_improved
```

### Train Model
```bash
python scripts/run_training.py --experiment v3_improved
```

### Walk-Forward Validation
```bash
python scripts/run_walk_forward_validation.py \
    --experiment v3_improved \
    --rolling-window-days 1095  # 3-year rolling
```

---

## 🏗️ Architecture

```
IntentFlowAI/
├── dashboard/app.py              # Streamlit trader dashboard
├── config/experiments/
│   ├── v_universe_full.yaml      # V2 configuration
│   └── v3_improved.yaml          # V3 configuration (latest)
├── intentflow_ai/
│   ├── features/
│   │   └── engineering.py        # 92 features (17 blocks)
│   ├── modeling/
│   │   ├── ensemble.py           # MultiAlgoEnsemble
│   │   ├── regimes.py            # 16-regime classifier
│   │   └── hmm_regime.py         # HMM regime detection (NEW)
│   └── monitoring/
│       └── ic_monitor.py         # IC monitoring & auto-retrain (NEW)
├── experiments/
│   ├── v_universe_full/          # V2 model artifacts
│   └── v3_improved/              # V3 model artifacts (latest)
└── scripts/
    ├── run_training.py           # Model training
    ├── run_scoring.py            # Generate predictions
    └── run_walk_forward_validation.py  # WFO testing
```

---

## 📈 Model Details

### MultiAlgoEnsemble
Combines 4 algorithms with regime-specific weighting:
- **LightGBM** (35%) - Fast gradient boosting
- **XGBoost** (30%) - Robust tree ensemble
- **CatBoost** (20%) - Categorical features
- **Ridge** (15%) - Linear regularization

### Feature Blocks (92 features)
| Block | Features | Status |
|-------|----------|--------|
| Technical | 6 | ✅ Active |
| Momentum | 10 | ✅ Active |
| Volatility | 5 | ✅ Active |
| Sector Relative | 6 | ✅ Active |
| Mean Reversion | 11 | ✅ Active |
| Macro | 14 | ✅ Active |
| Seasonality | 22 | ❌ Disabled (overfitting) |
| FII/DII Flow | 11 | ⏳ Pending data |

### Regime Classification
16 market regimes based on:
- Trend: bull/bear/sideways
- Volatility: high/medium/low
- Momentum: strong/weak

---

## 📦 Data Sources

| Data | Source | Status |
|------|--------|--------|
| **Price OHLCV** | Yahoo Finance | ✅ 15 years, 464 tickers |
| **Fundamentals** | EODHD | ✅ Quarterly reports |
| **Sectors** | yfinance | ✅ 462 tickers mapped |
| **VIX/Macro** | Yahoo Finance | ✅ Active |
| **FII/DII** | NSE | ❌ API limited |
| **Delivery %** | NSE | ❌ Slow fetching |

---

## 🔧 Recent Changes (V3)

### Completed
- [x] Disabled seasonality features (22 features causing overfitting)
- [x] Created HMM regime detector (`intentflow_ai/modeling/hmm_regime.py`)
- [x] Created IC monitoring system (`intentflow_ai/monitoring/ic_monitor.py`)
- [x] Switched to 3-year rolling training window
- [x] Increased regularization (L1=5.0, L2=20.0)

### Pending
- [ ] Integrate FII/DII data (NSE API limitations)
- [ ] Test shorter 2-year window
- [ ] Implement Transformer model component
- [ ] Add adaptive ensemble weighting

---

## ⚙️ Configuration

### V3 Config Highlights (`config/experiments/v3_improved.yaml`)
```yaml
splits:
  train_start: "2015-01-01"  # More recent data
  
wfo:
  rolling_window_years: 3    # Rolling, not expanding
  step_months: 3             # Quarterly rebalancing

trainer:
  params:
    max_depth: 3             # Reduced for regularization
    reg_lambda: 20.0         # Strong L2
    reg_alpha: 5.0           # Strong L1
```

---

## 📋 Known Issues

1. **IC Degradation**: Test IC near zero in 2024+
2. **Missing Data**: FII/DII, delivery % not available via free APIs
3. **Overfitting**: Train IC >> Test IC

---

## 📜 License

Proprietary - Internal use only.
