"""Streamlit dashboard entrypoint for IntentFlow AI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="IntentFlow AI", layout="wide")


@st.cache_data(show_spinner=False)
def load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["rank", "proba"], ascending=[True, False]).reset_index(drop=True)


st.title("IntentFlow AI Signal Monitor")
st.caption("Visualize ranked swing setups and filter by probability, recency, and sector.")

root = Path(__file__).resolve().parents[1]
latest_dir = root / "experiments" / "latest"
default_dir = latest_dir if latest_dir.exists() else root / "experiments" / "v0"
signals_path = default_dir / "top_signals.csv"

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh signals", use_container_width=True):
        with st.spinner("Running scoring pipeline..."):
            subprocess.run(["python", "scripts/run_scoring.py"], cwd=root, check=True)
            load_signals.clear()
            st.success("Signals refreshed.")
    file_path = st.text_input("Signals CSV path", value=str(signals_path))
    horizon_days = st.slider("Lookback window (days)", min_value=5, max_value=60, value=20, step=5)
    min_score = st.slider("Min probability", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

signals_path = Path(file_path).expanduser()
if not signals_path.exists():
    st.warning(f"No scoring file found at {signals_path}. Run the scoring script first.")
    st.stop()

signals = load_signals(signals_path)
if signals.empty:
    st.info("Signals file is empty. Populate it by running the scoring script.")
    st.stop()

latest_date = signals["date"].max()
cutoff = latest_date - pd.Timedelta(days=horizon_days)
filtered = signals[(signals["date"] >= cutoff) & (signals["proba"] >= min_score)].copy()
sector_options = sorted(filtered["sector"].dropna().unique())
selected_sectors = st.multiselect(
    "Sectors", options=sector_options, default=sector_options, placeholder="Select sectors"
)
if selected_sectors:
    filtered = filtered[filtered["sector"].isin(selected_sectors)]

st.subheader("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Latest date", latest_date.date().isoformat())
col2.metric("Signals shown", len(filtered))
mean_proba = filtered["proba"].mean() if not filtered.empty else 0.0
median_proba = filtered["proba"].median() if not filtered.empty else 0.0
col3.metric("Probability (mean / median)", f"{mean_proba:.2%} / {median_proba:.2%}")

st.subheader("Ranked signals")
if filtered.empty:
    st.info("No signals match the current filters. Adjust the sliders to broaden the view.")
else:
    styled = (
        filtered[["date", "ticker", "sector", "proba", "rank"]]
        .style.format({"proba": "{:.2%}", "rank": "{:.0f}"})
        .background_gradient(subset=["proba"], cmap="YlGn")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Top 10 probabilities")
    top10 = filtered.nsmallest(10, "rank")
    st.bar_chart(top10.set_index("ticker")[["proba"]])

st.subheader("Backtest overview")
summary_path = default_dir / "bt_summary.json"
equity_path = default_dir / "bt_equity.csv"
metric_cols = st.columns(3)
if summary_path.exists():
    summary = pd.read_json(summary_path, typ="series")
    metric_cols[0].metric("CAGR", f"{summary.get('CAGR', 0.0)*100:.2f}%")
    metric_cols[1].metric("Sharpe", f"{summary.get('Sharpe', 0.0):.2f}")
    metric_cols[2].metric("Max DD", f"{summary.get('maxDD', 0.0)*100:.2f}%")
else:
    st.info(f"Backtest summary not found at {summary_path}")

if equity_path.exists():
    equity = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    st.line_chart(equity.rename(columns={"equity": "PnL"}))
else:
    st.info(f"Backtest equity curve not found at {equity_path}")
