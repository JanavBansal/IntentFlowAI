# IntentFlow AI

A **production-ready systematic trading signal platform** for **NIFTY 500** (464 tickers) using a **multi-agent Council of Experts** architecture with LLM-powered debate synthesis.

---

## 📊 Current Status (Dec 2025)

### V4.5 Council of Experts (NEW)

| Agent | Model | Status |
|-------|-------|--------|
| **Technical Analyst** | LightGBM | ✅ Active |
| **Flow Detective** | XGBoost | ⏳ Needs delivery data |
| **Regime Sentinel** | 4-state HMM | ✅ Active |
| **Risk Contrarian** | Isolation Forest | ✅ Active |
| **Earnings Oracle** | Logistic Reg + EODHD | ✅ Active |

### Performance History

| Version | Architecture | Test IC | Status |
|---------|--------------|---------|--------|
| **V4.5** | Council of Experts | TBD | 🚧 New |
| V3 | Monolithic LightGBM | -0.001 | ⚠️ Alpha decay |
| V2 | Ensemble | 0.062 | Previous |

> **V4.5 Goal**: Address V3's alpha decay by combining specialized agents with debate-based synthesis and risk veto.

---

## 🚀 Quick Start

### Run Dashboard
```bash
streamlit run dashboard/app.py
```

### Train V4.5 Council
```python
from intentflow_ai.agents import CouncilOfExperts

council = CouncilOfExperts()
council.train_all_agents(X_train, y_train)
result = council.get_signal("RELIANCE", features)
```

### Test V4.5
```bash
python scripts/test_council.py --sample-size 5000
```

---

## 🏗️ Architecture

### V4.5 Council of Experts
```
                    ┌─────────────────┐
                    │  Input Features │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Technical       │ │ Flow Detective  │ │ Earnings Oracle │
│ Analyst (LGBM)  │ │ (XGBoost)       │ │ (LogReg+EODHD)  │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Regime Sentinel │ (4-state HMM)
                    │  → Agent Weights│
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Debate Protocol │ (LLM Synthesis)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Risk Contrarian │ (Veto Power)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Final Signal   │
                    └─────────────────┘
```

### Directory Structure
```
IntentFlowAI/
├── intentflow_ai/
│   ├── agents/                    # V4.5 Council (NEW)
│   │   ├── technical_analyst.py   # LightGBM wrapper
│   │   ├── flow_detective.py      # XGBoost delivery
│   │   ├── regime_sentinel.py     # 4-state HMM
│   │   ├── risk_contrarian.py     # Isolation Forest + veto
│   │   ├── earnings_oracle.py     # EODHD fundamentals
│   │   ├── debate_protocol.py     # LLM synthesis
│   │   └── council_workflow.py    # Main orchestration
│   ├── features/
│   │   ├── engineering.py         # 92 technical features
│   │   ├── delivery_features.py   # NEW: 15 delivery features
│   │   └── fundamental_features.py # NEW: EODHD merge
│   └── modeling/
│       ├── ensemble.py            # V3 legacy ensemble
│       └── hmm_regime.py          # HMM regime detection
├── config/experiments/
│   ├── v45_council.yaml           # V4.5 config (NEW)
│   └── v3_improved.yaml           # V3 config
└── scripts/
    ├── test_council.py            # V4.5 tests (NEW)
    └── fetch_historical_delivery_data.py  # Delivery fetch (NEW)
```

---

## 📦 Data Sources

| Data | Source | Records | Status |
|------|--------|---------|--------|
| **Price OHLCV** | Yahoo Finance | 15 years | ✅ Ready |
| **Fundamentals** | EODHD | 33K records, 468 tickers | ✅ Ready |
| **Sectors** | yfinance | 462 tickers | ✅ Ready |
| **Delivery %** | NSE/jugaad-data | — | ⏳ Script ready |
| **FII/DII** | NSE | — | ⏳ Pending |

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
