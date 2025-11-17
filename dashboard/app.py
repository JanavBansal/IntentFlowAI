"""Streamlit dashboard entrypoint for IntentFlow AI.

Enhanced with rolling IC, exposure metrics, feature drift detection, and SHAP explanations.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IntentFlow AI", layout="wide", page_icon="📊")


@st.cache_data(show_spinner=False)
def load_signals(path: Path) -> pd.DataFrame:
    """Load and parse signals CSV."""
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["rank", "proba"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_backtest_data(exp_dir: Path) -> Dict:
    """Load backtest summary and equity curve."""
    summary_path = exp_dir / "bt_summary.json"
    equity_path = exp_dir / "bt_equity.csv"
    trades_path = exp_dir / "bt_trades.csv"
    metrics_path = exp_dir / "metrics.json"
    
    data = {}
    if summary_path.exists():
        data["summary"] = json.loads(summary_path.read_text())
    if equity_path.exists():
        data["equity"] = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    if trades_path.exists():
        data["trades"] = pd.read_csv(trades_path, parse_dates=["date_in", "date_out"])
    if metrics_path.exists():
        data["metrics"] = json.loads(metrics_path.read_text())
    return data


def compute_rolling_ic(preds: pd.DataFrame, window: int = 30) -> pd.Series:
    """Compute rolling Information Coefficient."""
    if "excess_fwd" not in preds.columns or "proba" not in preds.columns:
        return pd.Series(dtype=float)
    
    preds = preds.sort_values("date")
    rolling_ic = []
    dates = []
    
    for i in range(window, len(preds)):
        window_data = preds.iloc[i-window:i]
        if len(window_data) < 10:
            continue
        ic = window_data["proba"].corr(window_data["excess_fwd"])
        if not np.isnan(ic):
            rolling_ic.append(ic)
            dates.append(window_data["date"].iloc[-1])
    
    return pd.Series(rolling_ic, index=pd.DatetimeIndex(dates), name="rolling_ic")


def compute_exposure_metrics(signals: pd.DataFrame, trades: pd.DataFrame | None = None) -> Dict:
    """Compute exposure and position metrics."""
    metrics = {}
    
    if not signals.empty:
        signals_by_date = signals.groupby("date")
        metrics["avg_daily_positions"] = signals_by_date.size().mean()
        metrics["max_daily_positions"] = signals_by_date.size().max()
        metrics["exposure_by_sector"] = signals.groupby("sector").size().to_dict() if "sector" in signals.columns else {}
    
    if trades is not None and not trades.empty:
        metrics["avg_hold_days"] = (trades["date_out"] - trades["date_in"]).dt.days.mean()
        metrics["active_positions"] = len(trades[trades["date_out"].isna()]) if "date_out" in trades.columns else 0
    
    return metrics


def detect_feature_drift(current_features: pd.DataFrame, reference_features: pd.DataFrame) -> pd.DataFrame:
    """Detect feature drift by comparing distributions."""
    if current_features.empty or reference_features.empty:
        return pd.DataFrame()
    
    drift_scores = []
    common_features = set(current_features.columns) & set(reference_features.columns)
    
    for feat in common_features:
        curr = current_features[feat].dropna()
        ref = reference_features[feat].dropna()
        
        if len(curr) < 5 or len(ref) < 5:
            continue
        
        # Kolmogorov-Smirnov test statistic (simplified)
        from scipy import stats
        try:
            ks_stat, _ = stats.ks_2samp(curr, ref)
            drift_scores.append({
                "feature": feat,
                "ks_statistic": ks_stat,
                "current_mean": curr.mean(),
                "reference_mean": ref.mean(),
                "current_std": curr.std(),
                "reference_std": ref.std(),
                "drift_severity": "high" if ks_stat > 0.3 else "medium" if ks_stat > 0.2 else "low"
            })
        except:
            pass
    
    return pd.DataFrame(drift_scores).sort_values("ks_statistic", ascending=False)


st.title("📊 IntentFlow AI - Live Trading Signal Dashboard")
st.caption("Real-time position-level signals with SHAP explanations, risk metrics, and performance monitoring")

root = Path(__file__).resolve().parents[1]
latest_dir = root / "experiments" / "latest"
default_dir = latest_dir if latest_dir.exists() else root / "experiments" / "v_universe_sanity"
signals_path = default_dir / "top_signals.csv"

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Refresh Signals", use_container_width=True):
        with st.spinner("Running scoring pipeline..."):
            try:
                subprocess.run(["python", "scripts/run_scoring.py", "--experiment", default_dir.name], 
                             cwd=root, check=True, capture_output=True)
                load_signals.clear()
                load_backtest_data.clear()
                st.success("✅ Signals refreshed!")
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
    
    file_path = st.text_input("Signals CSV path", value=str(signals_path))
    horizon_days = st.slider("Lookback window (days)", min_value=5, max_value=60, value=20, step=5)
    min_score = st.slider("Min probability", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    show_explanations = st.checkbox("Show SHAP explanations", value=True)

signals_path = Path(file_path).expanduser()
if not signals_path.exists():
    st.warning(f"⚠️ No scoring file found at {signals_path}. Run the scoring script first.")
    st.stop()

signals = load_signals(signals_path)
if signals.empty:
    st.info("📭 Signals file is empty. Populate it by running the scoring script.")
    st.stop()

latest_date = signals["date"].max()
cutoff = latest_date - pd.Timedelta(days=horizon_days)
filtered = signals[(signals["date"] >= cutoff) & (signals["proba"] >= min_score)].copy()

sector_options = sorted(filtered["sector"].dropna().unique()) if "sector" in filtered.columns else []
selected_sectors = st.multiselect(
    "Sectors", options=sector_options, default=sector_options, placeholder="Select sectors"
)
if selected_sectors and "sector" in filtered.columns:
    filtered = filtered[filtered["sector"].isin(selected_sectors)]

# === HEADLINE METRICS ===
st.subheader("📈 Headline Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Date", latest_date.date().isoformat())
col2.metric("Active Signals", len(filtered))
mean_proba = filtered["proba"].mean() if not filtered.empty else 0.0
col3.metric("Avg Probability", f"{mean_proba:.2%}")
col4.metric("Top Signal", filtered["ticker"].iloc[0] if not filtered.empty else "N/A")
col5.metric("Top Prob", f"{filtered['proba'].iloc[0]:.2%}" if not filtered.empty else "0%")

# === BACKTEST PERFORMANCE ===
st.subheader("💰 Backtest Performance")
bt_data = load_backtest_data(default_dir)
bt_cols = st.columns(4)

if "summary" in bt_data:
    summary = bt_data["summary"]
    bt_cols[0].metric("CAGR", f"{summary.get('CAGR', 0.0):.1%}" if isinstance(summary.get('CAGR'), (int, float)) and summary.get('CAGR') < 10 else f"{summary.get('CAGR', 0.0):.0f}%")
    bt_cols[1].metric("Sharpe Ratio", f"{summary.get('Sharpe', 0.0):.2f}")
    bt_cols[2].metric("Max Drawdown", f"{summary.get('maxDD', 0.0):.1%}")
    bt_cols[3].metric("Win Rate", f"{summary.get('win_rate', 0.0):.1%}")
else:
    st.info("📊 Backtest summary not available. Run backtest to see performance metrics.")

if "equity" in bt_data:
    equity = bt_data["equity"]
    st.line_chart(equity.rename(columns={"equity": "Portfolio Equity"}))

# === ROLLING IC & EXPOSURE ===
st.subheader("📊 Risk & Performance Metrics")

metrics_col1, metrics_col2 = st.columns(2)

with metrics_col1:
    st.markdown("**Rolling Information Coefficient**")
    if "metrics" in bt_data and "preds.csv" in str(default_dir):
        try:
            preds = pd.read_csv(default_dir / "preds.csv", parse_dates=["date"])
            rolling_ic = compute_rolling_ic(preds, window=30)
            if not rolling_ic.empty:
                st.line_chart(rolling_ic)
                st.metric("Current IC", f"{rolling_ic.iloc[-1]:.3f}" if len(rolling_ic) > 0 else "N/A")
                st.metric("Avg IC (30d)", f"{rolling_ic.mean():.3f}")
            else:
                st.info("Insufficient data for rolling IC")
        except Exception as e:
            st.warning(f"Could not compute rolling IC: {e}")
    else:
        st.info("Load predictions to compute rolling IC")

with metrics_col2:
    st.markdown("**Exposure Metrics**")
    exposure = compute_exposure_metrics(filtered, bt_data.get("trades"))
    exp_col1, exp_col2 = st.columns(2)
    exp_col1.metric("Avg Daily Positions", f"{exposure.get('avg_daily_positions', 0):.1f}")
    exp_col2.metric("Max Positions", exposure.get("max_daily_positions", 0))
    if exposure.get("exposure_by_sector"):
        st.bar_chart(pd.Series(exposure["exposure_by_sector"]))

# === SIGNAL TABLE WITH EXPLANATIONS ===
st.subheader("🎯 Ranked Trading Signals")

if filtered.empty:
    st.info("No signals match the current filters. Adjust the sliders to broaden the view.")
else:
    display_cols = ["date", "ticker", "sector", "proba", "rank"]
    if show_explanations and "rationale" in filtered.columns:
        display_cols.append("rationale")
    
    display_df = filtered[display_cols].copy()
    styled = display_df.style.format({
        "proba": "{:.2%}",
        "rank": "{:.0f}"
    }).background_gradient(subset=["proba"], cmap="YlGn")
    
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    # SHAP explanations expander
    if show_explanations and "top_features" in filtered.columns:
        with st.expander("🔍 View Detailed SHAP Explanations", expanded=False):
            selected_ticker = st.selectbox("Select ticker", filtered["ticker"].unique()[:20])
            ticker_signals = filtered[filtered["ticker"] == selected_ticker].head(1)
            if not ticker_signals.empty:
                signal = ticker_signals.iloc[0]
                st.write(f"**{signal['ticker']}** - {signal['date'].date()}")
                st.write(f"**Probability:** {signal['proba']:.2%}")
                if "rationale" in signal:
                    st.write(f"**Rationale:** {signal['rationale']}")
                if "top_features" in signal and isinstance(signal["top_features"], list):
                    st.write("**Top Contributing Features:**")
                    features_df = pd.DataFrame(signal["top_features"])
                    if not features_df.empty:
                        st.dataframe(features_df[["feature", "value", "shap_contribution"]].style.format({
                            "value": "{:.4f}",
                            "shap_contribution": "{:.4f}"
                        }), use_container_width=True)

# === FEATURE DRIFT DETECTION ===
st.subheader("🔬 Feature Drift Detection")
if st.checkbox("Enable feature drift analysis"):
    try:
        # Load training data as reference
        train_path = default_dir / "train.parquet"
        if train_path.exists():
            train_frame = pd.read_parquet(train_path)
            # Simplified: use recent signals as current distribution
            # In production, would load actual feature values
            st.info("Feature drift analysis requires feature engineering pipeline integration.")
            st.caption("This would compare current feature distributions against training set.")
        else:
            st.warning("Training data not found. Cannot compute feature drift.")
    except Exception as e:
        st.error(f"Error in drift detection: {e}")

# === SECTOR BREAKDOWN ===
if "sector" in filtered.columns and not filtered.empty:
    st.subheader("📊 Sector Distribution")
    sector_counts = filtered.groupby("sector").size().sort_values(ascending=False)
    st.bar_chart(sector_counts)

st.markdown("---")
st.caption("**IntentFlow AI** - Systematic trading signals for NIFTY 200 universe | "
          "Targets: >50% ROC AUC, >50% win rate, <25% drawdown")
