"""Streamlit dashboard entrypoint for IntentFlow AI."""

import streamlit as st

st.set_page_config(page_title="IntentFlow AI", layout="wide")

st.title("IntentFlow AI Signal Monitor")
st.caption("Visualize swing setups across the NIFTY 200 universe")

with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Signal horizon (days)", min_value=5, max_value=20, value=10)
    min_score = st.slider("Min probability", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

st.info(
    "Dashboard scaffolding is ready. Hook up the scoring pipeline once models "
    "and data sources are in place to display ranked opportunities."
)
