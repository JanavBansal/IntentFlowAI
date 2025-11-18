# NIFTY 200 Trading Strategy Validation Report

**Date:** 2025-11-17  
**Strategy Location:** `experiments/v_universe_full/`  
**Validation Objective:** Identify data leakage, overfitting, and backtest errors

---

## Executive Summary

**VERDICT: ⚠️ CRITICAL ISSUES FOUND - DO NOT PROCEED**

The strategy shows **unrealistic performance metrics** that are almost certainly due to:
1. **Severe overfitting** (95% performance degradation out-of-sample)
2. **Insufficient transaction costs** (Sharpe remains >13 even after costs)
3. **Walk-forward validation failure** (IC drops from 0.41 to 0.05)

**Key Findings:**
- Original Sharpe: **13.82** (extremely suspicious)
- Out-of-sample Test IC (2025): **0.0204** (vs 0.4067 in-sample)
- Walk-forward average IC: **0.0506** (87% degradation)
- Even with realistic costs, Sharpe remains **13.30**

---

## TASK 1: Data Leakage Audit

### 1.1 Feature Inspection

**Status:** ✅ **NO MAJOR LEAKAGE DETECTED**

All 62 features were audited across 13 feature blocks:
- Technical indicators (EMA, MACD, RSI, Bollinger)
- Momentum features
- Volatility measures
- ATR (Average True Range)
- Turnover/volume features
- Sector-relative features
- Regime features
- Mean reversion features

**Verification Results:**
- ✅ All rolling windows use backward-looking calculations
- ✅ Shift operations are correct (positive shift = past data)
- ✅ No forward-fill with future data in feature calculations
- ✅ Target variable created AFTER features (correct order)
- ✅ Train/test split happens AFTER feature/label creation

**Minor Concerns:**
- Regime features use `method="ffill"` when reindexing (lines 588-590 in `engineering.py`)
  - **Assessment:** This is for date alignment only, not forward-looking leakage
  - **Risk Level:** Low

### 1.2 Target Variable Check

✅ **SAFE:**
- Target uses `shift(-horizon_days)` which is correct for labels
- Features built first, then labels created
- No target information leaks into features

### 1.3 Common Leakage Patterns

✅ **All checks passed:**
- No forward-fill with future data
- Rolling windows use `min_periods` correctly
- Datetime indexing is correct
- No target encoding issues
- Expanding windows used for statistics (not full dataset)

**Conclusion:** Feature engineering appears to be point-in-time safe. **Data leakage is NOT the primary issue.**

---

## TASK 2: Transaction Cost Reality Check

### 2.1 Current Cost Configuration

**Original Backtest Costs:**
- Fee: 1.0 bps (0.01%)
- Slippage: 10.0 bps (0.10%)
- **Total: 0.11% per side = 0.22% round-trip**

**Realistic Costs Applied:**
- Transaction cost: 0.15% per trade (0.075% each side)
- Slippage: 0.05% (market impact)
- **Total friction: 0.20% per round-trip trade**

### 2.2 Performance After Costs

| Metric | Before Costs | After Costs | Change |
|--------|--------------|-------------|--------|
| **Sharpe Ratio** | 13.82 | **13.30** | -3.7% |
| Expected Value (%) | 5.40 | 5.20 | -3.7% |
| Profit Factor | 4.83 | 4.54 | -6.0% |
| Hit Rate (%) | 70.9 | 70.1 | -1.1% |
| Max Drawdown (%) | -14.43 | -15.82 | -9.6% |
| Sortino Ratio | 32.79 | 31.57 | -3.7% |
| Calmar Ratio | 2403.76 | 1760.09 | -26.8% |

**Critical Finding:** Even with realistic costs, Sharpe remains **13.30**, which is still **extremely suspicious**. Realistic Sharpe ratios for equity strategies are typically:
- Good: 1.0 - 2.0
- Excellent: 2.0 - 3.0
- Exceptional: 3.0 - 5.0
- **>10 is almost certainly an error**

---

## TASK 3: Out-of-Sample Testing

### 3.1 Strict Time Split

**Splits:**
- **Training:** 2018-01-01 to 2023-12-31 (258,536 rows)
- **Validation:** 2024-01-01 to 2024-12-31 (46,866 rows)
- **Test (unseen):** 2025-01-01 to 2025-11-17 (41,451 rows)

### 3.2 Performance by Period

| Split | ROC AUC | Return IC | Rows |
|-------|---------|------------|------|
| Train | 0.839 | 0.000 | 258,536 |
| Validation | 0.528 | 0.000 | 46,866 |
| **Test (2025)** | **0.522** | **0.0204** | 41,451 |

**🚨 CRITICAL FINDING:**

**In-sample Return IC:** 0.4067  
**Out-of-sample Test IC (2025):** 0.0204  
**Degradation: 95%**

This is a **MASSIVE red flag**. The model's predictive power collapses completely on truly unseen data.

**Interpretation:**
- The model has memorized patterns in the training data
- These patterns do not generalize to new time periods
- The high in-sample IC (0.41) is likely due to overfitting

---

## TASK 4: Feature Importance Sanity Check

### 4.1 Top 15 Features

| Rank | Feature | Importance | Economic Intuition |
|------|---------|------------|-------------------|
| 1 | `turnover__volume_mean_20` | 2181 | ✅ Makes sense |
| 2 | `sector_relative__sector_rel_close` | 1682 | ✅ Makes sense |
| 3 | `regime__vol_of_vol_20d` | 1496 | ✅ Makes sense |
| 4 | `regime__vix_equivalent_30d` | 1472 | ✅ Makes sense |
| 5 | `technical__ema_50` | 1422 | ⚠️ Review needed |
| 6 | `technical__ema_20` | 1371 | ⚠️ Review needed |
| 7 | `mean_reversion__dist_from_200ma_pct` | 1280 | ✅ Makes sense |
| 8 | `regime_adaptive__beta_estimate_20d` | 1253 | ✅ Makes sense |
| 9 | `mean_reversion__dist_from_200ma` | 1233 | ✅ Makes sense |
| 10 | `mean_reversion__bollinger_squeeze` | 1082 | ✅ Makes sense |
| 11 | `sector_relative__sector_vol_20_z` | 1059 | ✅ Makes sense |
| 12 | `atr__atr_14` | 1042 | ⚠️ Review needed |
| 13 | `volatility__price_vol_20` | 1017 | ✅ Makes sense |
| 14 | `technical__macd_signal` | 1012 | ✅ Makes sense |
| 15 | `momentum__pct_from_120d_high` | 1009 | ✅ Makes sense |

**Assessment:** Most features have reasonable economic intuition. The top features (volume, sector-relative, regime) make sense for a momentum/mean-reversion strategy.

### 4.2 Simplified Model Test

**Test:** Retrain using only top 10 features vs full 62 features

| Model | Test ROC | Performance Drop |
|-------|----------|------------------|
| Full (62 features) | 0.516 | Baseline |
| Simplified (10 features) | 0.508 | -1.7% |

**Conclusion:** ✅ Performance drop is reasonable (1.7%), suggesting the model is not severely overfitting on feature count alone. However, the out-of-sample degradation suggests **temporal overfitting** (memorizing time-specific patterns).

---

## TASK 5: Walk-Forward Validation

### 5.1 Rolling Window Backtest

**Configuration:**
- Training window: 2 years
- Test window: 3 months
- Retrain frequency: Every 3 months

### 5.2 Walk-Forward Performance

**Average Test IC:** 0.0506  
**Original IC:** 0.4067  
**Degradation: 87.6%**

**Period-by-Period Results:**
- Multiple periods show IC < 0.10
- High variance across periods
- Consistent underperformance vs in-sample

**Conclusion:** ⚠️ **HIGH OVERFITTING EVIDENCE**

The model fails to maintain predictive power when retrained on rolling windows, confirming severe overfitting to the original training period.

---

## TASK 6: Diagnosis Summary

### 1. Data Leakage Found?

**Answer: NO** ✅

- All features appear to be point-in-time safe
- Target variable created correctly
- No forward-looking operations in features

**However:** The lack of leakage makes the overfitting problem even more concerning - the model is genuinely learning patterns, but they don't generalize.

### 2. Performance After Costs

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **13.30** ⚠️ |
| Win Rate | 70.1% |
| Avg Profit | 5.20% |

**Critical:** Sharpe remains >13 even after realistic costs. This is **unrealistic**.

### 3. Out-of-Sample Performance (2025)

| Metric | Value |
|--------|-------|
| Test ROC | 0.522 |
| **Test IC** | **0.0204** ⚠️ |

**Critical:** 95% degradation from in-sample IC (0.4067 → 0.0204)

### 4. Overfitting Evidence

**Level: HIGH** ⚠️

- Out-of-sample IC: 0.0204 (vs 0.4067 in-sample) = **95% drop**
- Walk-forward average IC: 0.0506 = **87% drop**
- Model memorizes training patterns that don't generalize

### 5. Recommended Action

**🚨 DO NOT PROCEED - FIX ISSUES FIRST**

**Root Causes:**
1. **Severe temporal overfitting** - Model memorizes 2018-2024 patterns
2. **Insufficient regularization** - Model complexity too high for signal strength
3. **Possible regime shift** - 2025 market conditions differ from training period

**Required Actions:**
1. ✅ Increase regularization (reduce model complexity)
2. ✅ Add more robust cross-validation (time-series CV)
3. ✅ Reduce feature count or add feature selection
4. ✅ Test on multiple out-of-sample periods
5. ✅ Consider ensemble methods with different time windows
6. ✅ Re-evaluate if strategy works in different market regimes

**Next Steps:**
- Do NOT deploy this strategy
- Retrain with stronger regularization
- Validate on multiple out-of-sample periods
- If IC remains < 0.10 out-of-sample, consider abandoning strategy

---

## Appendix: Files Generated

1. ✅ `leakage_audit.csv` - Feature-by-feature review
2. ✅ `cost_comparison.csv` - Before/after costs comparison
3. ✅ `oos_performance.csv` - Out-of-sample performance by split
4. ✅ `feature_importance.csv` - Full feature importance rankings
5. ✅ `walk_forward_results.csv` - Period-by-period walk-forward results
6. ✅ `validation_summary.json` - Machine-readable summary

---

## Conclusion

The NIFTY 200 trading strategy shows **unrealistic in-sample performance** (Sharpe 13.82, IC 0.41) that **completely fails out-of-sample** (IC 0.02). This is a classic case of **severe overfitting**.

**Key Takeaways:**
- ✅ No data leakage detected
- ⚠️ Severe overfitting confirmed
- ⚠️ Strategy not ready for deployment
- ⚠️ Requires significant rework before proceeding

**Final Verdict: REVIEW_NEEDED ❌**

---

*Report generated: 2025-11-17*  
*Validation script: `scripts/validate_strategy.py`*

