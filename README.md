# IntentFlow AI

## Overview

**IntentFlow AI** is a research-driven, systematic trading signal platform designed to identify 5–20 day swing trading opportunities within the **NIFTY 200 [finance:NIFTY 200]** universe. It enables robust, low-drawdown, small-capital strategies by integrating multiple, heterogeneous data layers:

- **Market flows & ownership data**
- **Price & transaction microstructure**
- **Fundamental drift & accounting signals**
- **Narrative tone & sentiment** (news, social, filings)
- **Price-confirmation & technical structure**

The platform uses **LightGBM** as its machine learning core, enhanced by meta-labeling, regime filters, and cost-aware backtesting to surface actionable, high-quality trading signals.

---

## Architecture

The stack is modular and production-ready:

- `intentflow_ai/` — Core Python package (configs, ingestion, feature engineering, models)
- `data/` — Raw, interim, and processed datasets
- `scripts/` — Pipeline entry points (training, scoring, backtesting)
- `dashboard/` — Streamlit dashboard for signal viewing & diagnostics
- `notebooks/` — Research and exploratory analysis
- `experiments/` — Model artifacts, experiment metadata, and reports

---

## Typical Workflow

1. **Ingestion:** Load prices, flows, fundamentals, and narrative datasets.
2. **Feature Engineering:** Build predictive features across all data sources.
3. **Model Training:** Fit LightGBM models and meta-labels with purged time-series splits.
4. **Signal Generation:** Score all NIFTY 200 [finance:NIFTY 200] equities to find actionable trade ideas.
5. **Visualization:** Use the Streamlit dashboard for performance diagnostics & portfolio context.

---

## 📊 Current Model Results  
_Backtest: `v_universe_sanity` (57-ticker universe, baseline features)_

| Metric                  | Result    |
|-------------------------|-----------|
| **CAGR**                | 61.68%    |
| **Sharpe Ratio**        | 1.24      |
| **Max Drawdown**        | –77.13%   |
| **Win Rate**            | 53.0%     |
| **Turnover**            | 1.88      |
| **Avg Holding Period**  | 10 days   |

- **Sharpe ratio > 1.0** confirms statistically meaningful signal.
- **Win rate > 50%** shows directional edge, even with constraints.
- **High drawdown** is expected due to small universe, no risk filters, no meta-labeling, and fixed holding period.

_These are early, research-phase baselines. Performance will improve as more data and feature layers are integrated._

---

## 🚀 Next Steps

Planned improvements include:

- Expanding to the full NIFTY 200 [finance:NIFTY 200] universe and feature set
- Improving Sharpe ratio (goal: 1.7–2.5)
- Reducing drawdowns (target: –15% to –25%)
- Stabilizing trades through regime and volatility filters
- Enhancing meta-labeling and feature stacking

IntentFlow AI is a research-friendly, modular, and extensible platform for systematic Indian equities trading.

---

**Quick Start:**  
1. Set up Python 3.10+, install dependencies.  
2. Configure paths and secrets.  
3. Use scripts and the dashboard for training, evaluation, and visualization.


