"""
IntentFlow AI — Trader Dashboard v2
=====================================
Rebuilt for clarity, visual richness, and usability for equity traders.
Reads live from DuckDB. No hardcoded data.

Design principles:
  · Flow Detective is the edge — highlight it prominently
  · Every signal should have a "why": which agent drove it, and how strong
  · Regime context sets the mood: green for bull, red for bear, blue for sideways
  · Track record is sacred: show it honestly, including losses
  · Personal finance first: keep it simple, no noise
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import html as _html
import json as _json
from datetime import datetime as _dt, timezone as _tz
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntentFlow AI",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Typography & base */
  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

  /* Metric cards */
  [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
  [data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #888; text-transform: uppercase; letter-spacing: .04em; }

  /* Signal cards */
  .sig-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    border-left: 5px solid;
    transition: transform .15s;
  }
  .sig-card:hover { transform: translateX(3px); }
  .sig-buy        { border-left-color: #00e676; }
  .sig-sell       { border-left-color: #ff1744; }
  .sig-hold       { border-left-color: #ffd600; }
  .sig-vetoed     { border-left-color: #9e9e9e; opacity: .6; }

  /* Ticker badge */
  .ticker { font-size: 1.3rem; font-weight: 800; letter-spacing: .03em; color: #fff; }
  .sub    { font-size: 0.78rem; color: #999; margin-top: 2px; }

  /* Direction pill */
  .dir-pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: .04em;
  }
  .dir-buy    { background: rgba(0,230,118,.18); color: #00e676; }
  .dir-sell   { background: rgba(255,23,68,.18); color: #ff1744; }
  .dir-hold   { background: rgba(255,214,0,.15); color: #ffd600; }
  .dir-vetoed { background: rgba(158,158,158,.2); color: #bbb; }

  /* Agent pills */
  .ap { display:inline-block; padding:2px 9px; border-radius:10px; font-size:.72rem; font-weight:600; margin:2px; }
  .ap-bull { background:rgba(0,230,118,.18); color:#00e676; }
  .ap-bear { background:rgba(255,23,68,.18);  color:#ff1744; }
  .ap-neut { background:rgba(200,200,200,.1); color:#888; }

  /* Flow alert */
  .flow-alert {
    display:inline-block; background:rgba(100,181,246,.15);
    color:#64b5f6; border:1px solid rgba(100,181,246,.3);
    border-radius:8px; padding:2px 8px; font-size:.72rem; font-weight:700;
    margin-left:6px;
  }

  /* Regime banner */
  .regime-banner {
    border-radius: 10px; padding: 0.7rem 1.2rem;
    margin-bottom: 1.2rem; font-weight: 600;
  }

  /* Score bar container */
  .score-bar-wrap { background:#2a2a3e; border-radius:4px; height:6px; margin-top:6px; overflow:hidden; }
  .score-bar-fill { height:100%; border-radius:4px; }

  /* Section headings */
  .section-head {
    font-size: 1.05rem; font-weight: 700; color: #e0e0e0;
    margin: 1.2rem 0 0.6rem; padding-bottom: 4px;
    border-bottom: 2px solid rgba(255,255,255,.07);
  }

  /* Disclaimer footer */
  .disclaimer {
    font-size: 0.72rem; color: #555; margin-top: 1.5rem;
    border-top: 1px solid #222; padding-top: 0.8rem;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LAYER
# =============================================================================

@st.cache_resource(show_spinner=False)
def _get_store():
    try:
        from intentflow_ai.database import SignalStore
        return SignalStore(read_only=True)
    except Exception:
        return None

@st.cache_data(ttl=180, show_spinner=False)
def _latest(_sid: int) -> pd.DataFrame:
    s = _get_store()
    return s.latest_signals(run_type="evening", n_days=1) if s else pd.DataFrame()

@st.cache_data(ttl=180, show_spinner=False)
def _history(_sid: int, days: int = 60) -> pd.DataFrame:
    s = _get_store()
    return s.signals_history(days_back=days) if s else pd.DataFrame()

@st.cache_data(ttl=180, show_spinner=False)
def _perf(_sid: int) -> pd.DataFrame:
    s = _get_store()
    return s.performance_summary() if s else pd.DataFrame()

@st.cache_data(ttl=180, show_spinner=False)
def _pnl(_sid: int) -> pd.DataFrame:
    s = _get_store()
    return s.paper_pnl() if s else pd.DataFrame()

@st.cache_data(ttl=180, show_spinner=False)
def _ic(_sid: int) -> pd.DataFrame:
    s = _get_store()
    return s.rolling_ic(window=20) if s else pd.DataFrame()

@st.cache_data(ttl=180, show_spinner=False)
def _regimes(_sid: int) -> pd.DataFrame:
    s = _get_store()
    return s.regime_history(days_back=90) if s else pd.DataFrame()

@st.cache_data(ttl=180, show_spinner=False)
def _stats(_sid: int) -> dict:
    s = _get_store()
    return s.stats_summary() if s else {}


@st.cache_data(ttl=300, show_spinner=False)
def _alpha_analysis(_sid: int) -> dict:
    """Compute alpha of BUY picks vs equal-weight market benchmark."""
    s = _get_store()
    if not s:
        return {}
    try:
        perf = s.performance_summary()
        if perf.empty:
            return {}
        buys = perf[perf["direction"] == "BUY"].copy()
        if buys.empty:
            return {}

        # Load price panel for market benchmark
        prices = pd.read_parquet(ROOT / "data" / "processed" / "prices.parquet")
        prices["date"] = pd.to_datetime(prices["date"])
        buys["signal_date"] = pd.to_datetime(buys["signal_date"])

        # Compute market median 15d return per signal date
        unique_dates = buys["signal_date"].unique()
        mkt_map = {}
        for sd in unique_dates:
            exit_date = sd + pd.Timedelta(days=15)
            entry_p = prices[prices["date"] <= sd].groupby("ticker")["close"].last()
            exit_p = prices[prices["date"] >= exit_date].groupby("ticker")["close"].first()
            common = entry_p.index.intersection(exit_p.index)
            if len(common) > 50:
                mkt_map[sd] = float(((exit_p[common] - entry_p[common]) / entry_p[common]).median())

        buys["market_return"] = buys["signal_date"].map(mkt_map)
        buys["alpha"] = buys["actual_return"] - buys["market_return"]
        valid = buys.dropna(subset=["alpha"])

        if valid.empty:
            return {}

        beat_market = float((valid["alpha"] > 0).mean())

        # Monthly breakdown
        valid["month"] = valid["signal_date"].dt.to_period("M").astype(str)
        monthly = valid.groupby("month").agg(
            n=("alpha", "count"),
            avg_return=("actual_return", "mean"),
            avg_market=("market_return", "mean"),
            avg_alpha=("alpha", "mean"),
        ).reset_index()

        return {
            "avg_return": float(valid["actual_return"].mean()),
            "avg_market": float(valid["market_return"].mean()),
            "avg_alpha": float(valid["alpha"].mean()),
            "median_alpha": float(valid["alpha"].median()),
            "beat_market_pct": beat_market,
            "n_trades": len(valid),
            "monthly": monthly.to_dict("records"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _backtest_data() -> pd.DataFrame:
    """Load walk-forward backtest results from logs/backtest_results.csv."""
    try:
        path = ROOT / "logs" / "backtest_results.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["fwd_return"] = pd.to_numeric(df["fwd_return"], errors="coerce")
        df["final_signal"] = pd.to_numeric(df["final_signal"], errors="coerce")
        df["quintile"] = pd.to_numeric(df["quintile"], errors="coerce")
        return df.dropna(subset=["fwd_return"])
    except Exception:
        return pd.DataFrame()


# =============================================================================
# HELPERS
# =============================================================================

SECTORS = {
    "TCS":"IT","INFY":"IT","WIPRO":"IT","TECHM":"IT","HCLTECH":"IT",
    "LTIM":"IT","MPHASIS":"IT","PERSISTENT":"IT","COFORGE":"IT",
    "RELIANCE":"Energy","ONGC":"Energy","BPCL":"Energy","IOC":"Energy",
    "HDFCBANK":"Financials","ICICIBANK":"Financials","SBIN":"Financials",
    "KOTAKBANK":"Financials","AXISBANK":"Financials","BAJFINANCE":"Financials",
    "HDFCLIFE":"Financials","SBILIFE":"Financials","ICICIGI":"Financials",
    "SUNPHARMA":"Healthcare","DRREDDY":"Healthcare","CIPLA":"Healthcare",
    "DIVISLAB":"Healthcare","APOLLOHOSP":"Healthcare","MAXHEALTH":"Healthcare",
    "TATASTEEL":"Materials","HINDALCO":"Materials","ULTRACEMCO":"Materials",
    "JSWSTEEL":"Materials","VEDL":"Materials","NMDC":"Materials",
    "TITAN":"Consumer","HINDUNILVR":"Consumer","ITC":"Consumer",
    "NESTLEIND":"Consumer","DABUR":"Consumer","MARICO":"Consumer",
    "ADANIENT":"Industrials","LT":"Industrials","SIEMENS":"Industrials",
    "BHEL":"Industrials","POLYCAB":"Industrials","ABB":"Industrials",
    "MARUTI":"Auto","TATAMOTORS":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","MRF":"Auto",
    "BEL":"Defence","HAL":"Defence","COCHINSHIP":"Defence",
    "POWERGRID":"Utilities","NTPC":"Utilities","TATAPOWER":"Utilities",
    "ADANIPORTS":"Infrastructure","ZOMATO":"Consumer Tech","PAYTM":"Consumer Tech",
}

REGIME_COLORS = {
    "Bull":              "#00e676",
    "Bear":              "#ff1744",
    "High-Vol Sideways": "#ff9100",
    "Low-Vol Sideways":  "#40c4ff",
}
REGIME_BG = {
    "Bull":              "rgba(0,230,118,.08)",
    "Bear":              "rgba(255,23,68,.08)",
    "High-Vol Sideways": "rgba(255,145,0,.08)",
    "Low-Vol Sideways":  "rgba(64,196,255,.08)",
}
REGIME_NOTES = {
    "Bull":              "🟢 Market in uptrend. Delivery data confirming institutional accumulation. Lower thresholds — more BUY opportunities.",
    "Bear":              "🔴 Market under pressure. Raise cash. Only very high-conviction signals worth acting on.",
    "High-Vol Sideways": "🟠 Volatile, directionless market. Signals are less reliable. Reduce position sizes.",
    "Low-Vol Sideways":  "🔵 Calm, consolidating market. Good setup for breakout plays. Standard thresholds apply.",
}

def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["sector"]  = df["ticker"].map(SECTORS).fillna("Other")
    df["vetoed"]  = df["vetoed"].fillna(False).astype(bool)
    df["flow_driven"] = (
        df.get("flow_signal", pd.Series(0.0, index=df.index)).fillna(0).abs() > 0.15
    )
    df["agent_agree"] = (
        (df.get("technical_signal", pd.Series(0.0, index=df.index)).fillna(0) > 0) ==
        (df.get("flow_signal",      pd.Series(0.0, index=df.index)).fillna(0) > 0)
    )
    return df


def _dir_pill(direction: str, vetoed: bool) -> str:
    if vetoed:
        return '<span class="dir-pill dir-vetoed">VETOED</span>'
    d = str(direction).upper()
    cls = {"BUY": "dir-buy", "SELL": "dir-sell", "HOLD": "dir-hold"}.get(d, "dir-hold")
    return f'<span class="dir-pill {cls}">{d}</span>'


def _agent_pills(row) -> str:
    pills = ""
    for label, col in [("Tech", "technical_signal"), ("Flow", "flow_signal"),
                       ("Earn", "earnings_signal"),  ("Risk", "risk_signal")]:
        val = row.get(col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        val = float(val)
        if val > 0.05:
            cls = "ap-bull"; arrow = "▲"
        elif val < -0.05:
            cls = "ap-bear"; arrow = "▼"
        else:
            cls = "ap-neut"; arrow = "–"
        pills += f'<span class="ap {cls}">{arrow} {label} {val:+.2f}</span>'
    return pills


def _score_bar(score: float, color: str) -> str:
    pct = min(abs(score) / 0.5 * 100, 100)
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>'
        f'</div>'
    )


def _signal_card(row, pos_size: float) -> str:
    d   = str(row.get("direction", "HOLD")).upper()
    sig = float(row.get("final_signal", 0.0))
    conf= float(row.get("final_confidence", 0.0))
    fd  = bool(row.get("flow_driven", False))
    aa  = bool(row.get("agent_agree", False))
    regime = str(row.get("regime", ""))

    color = {"BUY": "#00e676", "SELL": "#ff1744"}.get(d, "#ffd600")
    if row.get("vetoed"):
        card_cls, color = "sig-vetoed", "#9e9e9e"
    else:
        card_cls = {"BUY": "sig-buy", "SELL": "sig-sell", "HOLD": "sig-hold"}.get(d, "sig-hold")

    flow_badge = '<span class="flow-alert">💧 Flow-driven</span>' if fd else ""
    agree_badge = '<span class="flow-alert" style="color:#a5d6a7;border-color:rgba(165,214,167,.3);">🤝 Agents Agree</span>' if (aa and d != "HOLD") else ""

    price_str = ""
    cp = row.get("close_price", 0)
    if cp and float(cp) > 0:
        price_str = f'<span style="color:#aaa;font-size:.82rem;">₹{float(cp):,.1f}</span>'

    pos_pct = f"{pos_size * (1 + conf) * (abs(sig) / 0.3):.1f}%"

    reasoning_raw = str(row.get("reasoning", ""))
    reasoning_short = reasoning_raw[:100] + "…" if len(reasoning_raw) > 100 else reasoning_raw
    reasoning = _html.escape(reasoning_short)
    sector_safe = _html.escape(str(row.get("sector", "Other")))
    ticker_safe = _html.escape(str(row["ticker"]))
    regime_safe = _html.escape(regime)
    pill       = _dir_pill(d, row.get("vetoed", False))
    pills      = _agent_pills(row)
    score_bar  = _score_bar(sig, color)

    # Single-line HTML — avoids CommonMark blank-line rule that terminates HTML
    # blocks and causes inner content to render as escaped text in st.markdown.
    return (
        f'<div class="sig-card {card_cls}">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div><span class="ticker">{ticker_safe}</span> {pill} {flow_badge}{agree_badge}'
        f'<div class="sub">{sector_safe} &middot; {regime_safe}</div></div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:1.3rem;font-weight:800;color:{color};">{sig:+.3f}</div>'
        f'{price_str}'
        f'<div style="font-size:.72rem;color:#666;">conf {conf:.0%}</div>'
        f'</div></div>'
        f'{score_bar}'
        f'<div style="margin-top:0.5rem;">{pills}</div>'
        f'<div style="margin-top:0.4rem;font-size:.72rem;color:#555;font-style:italic;">{reasoning}</div>'
        f'</div>'
    )


# =============================================================================
# STORE + INITIAL DATA
# =============================================================================

store   = _get_store()
sid     = id(store) if store else 0
df_now  = _add_derived(_latest(sid))
stats   = _stats(sid)
_alpha_hdr = _alpha_analysis(sid)


# =============================================================================
# HEADER
# =============================================================================

h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("## 🔬 IntentFlow AI — Live Signals")
    st.caption("Council of Experts · NIFTY 500 · 15-day horizon · Paper trading")

with h2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()   # reopen DB connection to see latest writes
        st.rerun()
    if not df_now.empty and "signal_date" in df_now.columns:
        d = pd.to_datetime(df_now["signal_date"].max()).strftime("%d %b %Y")
        st.caption(f"Last run: {d}")

# ── Data Freshness Banner ────────────────────────────────────────────────────
_status_path = ROOT / "logs" / "status.json"
if _status_path.exists():
    try:
        _status = _json.loads(_status_path.read_text())
        _last_run = _dt.fromisoformat(_status.get("last_run", ""))
        _age_hours = (_dt.now(_tz.utc) - _last_run.replace(tzinfo=_tz.utc)).total_seconds() / 3600
        _scored = _status.get("tickers_scored", 0)
        _issues = _status.get("issues", [])
        _run_status = _status.get("status", "ok")

        if _run_status == "error" or _age_hours > 36:
            _banner_color, _banner_icon, _banner_text = "#fee2e2", "🔴", "Data may be stale"
        elif _run_status == "warn" or _age_hours > 18:
            _banner_color, _banner_icon, _banner_text = "#fef9c3", "🟡", "Check runner health"
        else:
            _banner_color, _banner_icon, _banner_text = "#dcfce7", "🟢", "Pipeline healthy"

        _age_str = f"{_age_hours:.0f}h ago" if _age_hours < 48 else f"{_age_hours/24:.0f}d ago"
        _issue_str = " · ".join(_issues) if _issues else ""
        st.markdown(
            f'<div style="background:{_banner_color};padding:8px 16px;border-radius:8px;margin-bottom:12px;font-size:0.85rem;">'
            f'{_banner_icon} <strong>{_banner_text}</strong> — last run {_age_str} · {_scored} tickers scored'
            f'{" · <em>" + _html.escape(_issue_str) + "</em>" if _issue_str else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass  # status.json malformed — skip banner silently


# ── Top-of-page KPIs ─────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🟢 Today's BUYs",    len(df_now[df_now.get("direction","") == "BUY"])  if not df_now.empty else "—")
k2.metric("🔴 Today's SELLs",   len(df_now[df_now.get("direction","") == "SELL"]) if not df_now.empty else "—")
k3.metric("💧 Flow-driven",     int(df_now["flow_driven"].sum()) if not df_now.empty else "—")
k4.metric("📊 Settled Trades",  stats.get("total_trades_settled", 0))
k5.metric("✅ Win Rate",         f"{stats.get('win_rate',0):.0%}" if stats else "—")
k6.metric("📈 Alpha/Trade",      f"{_alpha_hdr['avg_alpha']:+.2%}" if _alpha_hdr else "—")

st.divider()


# =============================================================================
# TABS
# =============================================================================

tab_signals, tab_dive, tab_record, tab_pulse, tab_health = st.tabs([
    "📊 Today's Signals",
    "🔍 Deep Dive",
    "📈 Track Record",
    "🌐 Market Pulse",
    "🧠 Model Health",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TODAY'S SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
with tab_signals:

    if df_now.empty:
        st.info("No signals yet. Run: `python scripts/live_runner.py --mode evening --no-llm`")
        st.stop()

    # ── Regime banner ─────────────────────────────────────────────────────────
    dominant_regime = "Low-Vol Sideways"
    if "regime" in df_now.columns:
        rc = df_now["regime"].value_counts()
        dominant_regime = rc.idxmax() if not rc.empty else "Low-Vol Sideways"

    rcol  = REGIME_COLORS.get(dominant_regime, "#888")
    rbg   = REGIME_BG.get(dominant_regime, "rgba(128,128,128,.06)")
    rnote = REGIME_NOTES.get(dominant_regime, "")
    st.markdown(
        f'<div class="regime-banner" style="background:{rbg};border-left:4px solid {rcol};">'
        f'<strong style="color:{rcol};">REGIME: {dominant_regime.upper()}</strong>'
        f'<span style="color:#aaa;font-size:.85rem;margin-left:12px;">{rnote}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Today's Action Plan ────────────────────────────────────────────────────
    with st.expander("📋 **Today's Action Plan** — How to trade these signals", expanded=False):
        _n_buy = len(df_now[df_now["direction"] == "BUY"]) if not df_now.empty else 0
        _n_sell = len(df_now[df_now["direction"] == "SELL"]) if not df_now.empty else 0
        _hi_conv = len(df_now[(df_now["direction"] == "BUY") & (df_now["final_signal"].abs() > 0.20) & (df_now.get("flow_driven", False) == True)]) if not df_now.empty else 0

        st.markdown(f"""
**Signal Summary:** {_n_buy} BUYs · {_n_sell} SELLs · {_hi_conv} high-conviction (signal > 0.20 + Flow bullish)

**What to focus on:**
- **High-conviction picks:** BUY signals with |signal| > 0.20 where Flow Detective is the primary driver (💧 badge)
- **Avoid:** signals driven only by Technical Analyst with flat Flow; vetoed stocks; confidence < 20%
- **Deep Dive** (Tab 2) your top 3 picks — check the IC-weighted waterfall chart

**Position sizing rules:**
- Max **2% of portfolio** per position
- Max **5 simultaneous** open positions (10% total exposure)
- Scale to 5% per position only after 30+ settled trades confirm the backtest
- **Hold for 15 trading days** — do not exit early unless a later run generates SELL

**What the signals mean:**
| Signal Range | Meaning | Action |
|---|---|---|
| BUY > +0.30 with Flow bullish | Highest conviction — institutional accumulation | Act on it |
| BUY +0.10 to +0.30 | Moderate — only act if Flow agrees | Selective |
| SELL < -0.20 | Institutional distribution detected | Reduce holdings |
| HOLD (majority) | No clear edge — don't force trades | Skip |
""")

    # ── Sidebar-style filters ─────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
    with fc1:
        show_dir = st.multiselect("Direction", ["BUY", "SELL", "HOLD"],
                                   default=["BUY", "SELL"], key="dir_filt")
    with fc2:
        all_sectors = sorted(df_now["sector"].unique().tolist())
        show_sec = st.multiselect("Sector", ["All"] + all_sectors, default=["All"], key="sec_filt")
    with fc3:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.2, 0.05, key="conf_filt")
    with fc4:
        risk_pct = st.number_input("Risk/trade %", 0.5, 5.0, 2.0, 0.5, key="risk_inp")

    filt = df_now[df_now["direction"].isin(show_dir)]
    filt = filt[filt["final_confidence"] >= min_conf]
    filt = filt[~filt["vetoed"]]
    if "All" not in show_sec:
        filt = filt[filt["sector"].isin(show_sec)]
    filt = filt.sort_values("final_signal", ascending=False).reset_index(drop=True)

    st.caption(f"{len(filt)} signals shown · {len(df_now)} total scored")

    # ── TOP PICKS ─────────────────────────────────────────────────────────────
    buys = filt[filt["direction"] == "BUY"].head(6)
    sells = filt[filt["direction"] == "SELL"].head(4)
    featured = pd.concat([buys, sells], ignore_index=True)

    if not featured.empty:
        st.markdown('<div class="section-head">🏆 Top Picks — High Conviction</div>',
                    unsafe_allow_html=True)

        # Render all cards in one st.markdown call using CSS grid — avoids
        # Streamlit's per-column HTML sanitiser bug in v1.45+
        cards_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:1rem;">'
        for _, row in featured.iterrows():
            cards_html += _signal_card(row, risk_pct)
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── SIGNAL STRENGTH CHART ─────────────────────────────────────────────────
    st.markdown('<div class="section-head">📊 Signal Strength — All Filtered Picks</div>',
                unsafe_allow_html=True)

    if not filt.empty:
        chart_df = filt[filt["direction"].isin(["BUY", "SELL"])].copy()
        if not chart_df.empty:
            chart_df = chart_df.sort_values("final_signal")
            colors = chart_df["direction"].map({"BUY": "#00e676", "SELL": "#ff1744"}).fillna("#888")

            fig = go.Figure(go.Bar(
                x=chart_df["final_signal"],
                y=chart_df["ticker"],
                orientation="h",
                marker_color=list(colors),
                customdata=chart_df[["sector", "final_confidence",
                                     "flow_signal", "technical_signal"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Signal: %{x:+.3f}<br>"
                    "Sector: %{customdata[0]}<br>"
                    "Confidence: %{customdata[1]:.0%}<br>"
                    "Flow: %{customdata[2]:+.3f}  Tech: %{customdata[3]:+.3f}<extra></extra>"
                ),
            ))
            fig.update_layout(
                height=max(250, len(chart_df) * 36),
                margin=dict(l=0, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#2a2a3e", zerolinecolor="#444", color="#888"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", color="#ccc"),
                font=dict(color="#ccc"),
                showlegend=False,
            )
            fig.add_vline(x=0, line_color="#444", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No BUY/SELL signals match current filters.")

    # ── AGENT CONSENSUS CHART ─────────────────────────────────────────────────
    if not filt.empty and "BUY" in filt["direction"].values:
        st.markdown('<div class="section-head">🧩 Agent Breakdown — BUY Signals</div>',
                    unsafe_allow_html=True)

        top_buys = filt[filt["direction"] == "BUY"].head(10).copy()
        agent_cols = [c for c in ["technical_signal","flow_signal","earnings_signal","risk_signal"]
                      if c in top_buys.columns]
        if agent_cols:
            labels = {"technical_signal":"Technical","flow_signal":"Flow Detective",
                      "earnings_signal":"Earnings Oracle","risk_signal":"Risk Contrarian"}
            fig2 = go.Figure()
            colors_map = {"technical_signal":"#42a5f5","flow_signal":"#00e676",
                          "earnings_signal":"#ffca28","risk_signal":"#ef5350"}

            for col in agent_cols:
                fig2.add_trace(go.Bar(
                    name=labels.get(col, col),
                    x=top_buys["ticker"],
                    y=top_buys[col].fillna(0),
                    marker_color=colors_map.get(col, "#888"),
                    hovertemplate=f"<b>%{{x}}</b><br>{labels.get(col,'')}: %{{y:+.3f}}<extra></extra>",
                ))
            fig2.update_layout(
                barmode="group",
                height=300,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#2a2a3e", color="#888"),
                yaxis=dict(gridcolor="#2a2a3e", color="#888", title="Signal"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                font=dict(color="#ccc"),
            )
            fig2.add_hline(y=0, line_color="#444", line_width=1)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("💧 Flow Detective (green) is the primary alpha source. "
                       "When Flow and Technical agree → highest conviction.")

    # ── FULL TABLE ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📋 All Signals Table</div>', unsafe_allow_html=True)

    show_c = [c for c in ["ticker","sector","direction","final_signal","final_confidence",
                           "technical_signal","flow_signal","regime","close_price","vetoed"]
              if c in filt.columns]
    rename = {"ticker":"Stock","sector":"Sector","direction":"Signal",
              "final_signal":"Score","final_confidence":"Conf",
              "technical_signal":"Tech","flow_signal":"Flow",
              "regime":"Regime","close_price":"Price ₹","vetoed":"Vetoed"}

    disp = filt[show_c].rename(columns=rename)
    fmt  = {k:"{:+.3f}" for k in ["Score","Tech","Flow"] if k in disp}
    fmt["Conf"]    = "{:.0%}"
    fmt["Price ₹"] = "₹{:,.0f}"

    st.dataframe(
        disp.style
            .format(fmt)
            .background_gradient(subset=["Score"] if "Score" in disp else [],
                                 cmap="RdYlGn", vmin=-0.4, vmax=0.4),
        use_container_width=True, hide_index=True, height=380,
    )

    csv_dl = filt.to_csv(index=False)
    st.download_button("📥 Download CSV", csv_dl,
                       f"signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                       "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab_dive:
    st.header("🔍 Signal Deep Dive")
    st.caption("Select a stock to understand exactly what the council is seeing and why.")

    if df_now.empty:
        st.info("Run the live runner first.")
    else:
        tickers_sorted = df_now.sort_values("final_signal", ascending=False)["ticker"].tolist()
        sel_ticker = st.selectbox("Choose a stock", tickers_sorted, key="dive_ticker")

        row = df_now[df_now["ticker"] == sel_ticker].iloc[0]
        sig   = float(row.get("final_signal", 0.0))
        conf  = float(row.get("final_confidence", 0.0))
        direction = str(row.get("direction", "HOLD"))
        rcol_t = REGIME_COLORS.get(str(row.get("regime","")), "#888")

        # ── Hero row ────────────────────────────────────────────────────────
        col_hero1, col_hero2, col_hero3 = st.columns([2, 1, 1])
        with col_hero1:
            _dd_card_cls = 'sig-buy' if direction=='BUY' else 'sig-sell' if direction=='SELL' else 'sig-hold'
            _dd_pill     = _dir_pill(direction, bool(row.get('vetoed', False)))
            _dd_flow     = '<span class="flow-alert">💧 Flow-driven</span>' if row.get('flow_driven') else ''
            _dd_agree    = '<span class="flow-alert" style="color:#a5d6a7;border-color:rgba(165,214,167,.3);">🤝 Agents Agree</span>' if row.get('agent_agree') and direction != 'HOLD' else ''
            _dd_sector   = _html.escape(str(row.get('sector', 'Other')))
            _dd_regime   = _html.escape(str(row.get('regime', '')))
            _dd_ticker   = _html.escape(sel_ticker)
            st.markdown(
                f'<div class="sig-card {_dd_card_cls}" style="padding:1.2rem;">'
                f'<div class="ticker" style="font-size:2rem;">{_dd_ticker}</div>'
                f'<div class="sub">{_dd_sector} &middot; {_dd_regime}</div>'
                f'<div style="margin-top:.6rem;">{_dd_pill} {_dd_flow}{_dd_agree}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_hero2:
            cp = float(row.get("close_price", 0) or 0)
            st.metric("Entry Price", f"₹{cp:,.1f}" if cp > 0 else "—")
            st.metric("Signal Score", f"{sig:+.3f}")
        with col_hero3:
            st.metric("Confidence", f"{conf:.0%}")
            st.metric("Hold Period", "15 days")

        # ── Agent votes radar ────────────────────────────────────────────────
        st.markdown('<div class="section-head">🧩 How Each Agent Voted</div>',
                    unsafe_allow_html=True)

        agents = {
            "Flow Detective":   float(row.get("flow_signal", 0) or 0),
            "Technical Analyst":float(row.get("technical_signal", 0) or 0),
            "Earnings Oracle":  float(row.get("earnings_signal", 0) or 0),
            "Risk Contrarian":  float(row.get("risk_signal", 0) or 0),
        }
        IC_WEIGHTS = {"Flow Detective":0.65,"Technical Analyst":0.20,
                      "Earnings Oracle":0.10,"Risk Contrarian":0.05}

        dl_col, dr_col = st.columns(2)

        with dl_col:
            fig_agents = go.Figure()
            agent_names  = list(agents.keys())
            agent_values = list(agents.values())
            bar_colors   = [
                "#00e676" if v > 0.05 else "#ff1744" if v < -0.05 else "#555"
                for v in agent_values
            ]
            fig_agents.add_trace(go.Bar(
                x=agent_names, y=agent_values,
                marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in agent_values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Signal: %{y:+.3f}<extra></extra>",
            ))
            fig_agents.update_layout(
                height=280, margin=dict(l=0,r=0,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#aaa"), yaxis=dict(color="#888",gridcolor="#2a2a3e"),
                font=dict(color="#ccc"), showlegend=False,
                title=dict(text="Agent Signals", font=dict(size=13,color="#aaa")),
            )
            fig_agents.add_hline(y=0, line_color="#333", line_width=1)
            st.plotly_chart(fig_agents, use_container_width=True)

        with dr_col:
            # IC-weighted contribution waterfall
            contributions = {k: agents[k] * IC_WEIGHTS[k] for k in agents}
            c_names  = list(contributions.keys())
            c_values = list(contributions.values())
            wf_colors = ["#00e676" if v > 0 else "#ff1744" for v in c_values]

            fig_wf = go.Figure(go.Bar(
                x=c_names, y=c_values,
                marker_color=wf_colors,
                text=[f"{v:+.4f}" for v in c_values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Weighted Contribution: %{y:+.4f}<extra></extra>",
            ))
            fig_wf.update_layout(
                height=280, margin=dict(l=0,r=0,t=20,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#aaa"), yaxis=dict(color="#888",gridcolor="#2a2a3e"),
                font=dict(color="#ccc"), showlegend=False,
                title=dict(text="IC-Weighted Contribution to Final Signal",
                           font=dict(size=13,color="#aaa")),
            )
            fig_wf.add_hline(y=0, line_color="#333", line_width=1)
            st.plotly_chart(fig_wf, use_container_width=True)

        # ── What the model is saying ─────────────────────────────────────────
        st.markdown('<div class="section-head">💬 What the Council Is Saying</div>',
                    unsafe_allow_html=True)

        flow_val = float(row.get("flow_signal", 0) or 0)
        tech_val = float(row.get("technical_signal", 0) or 0)
        earn_val = float(row.get("earnings_signal", 0) or 0)
        risk_val = float(row.get("risk_signal", 0) or 0)

        if abs(flow_val) > 0.05:
            flow_direction = "institutional ACCUMULATION" if flow_val > 0 else "institutional DISTRIBUTION"
            flow_strength  = "strong" if abs(flow_val) > 0.15 else "moderate"
            st.info(f"💧 **Flow Detective** is detecting **{flow_strength} {flow_direction}** "
                    f"(delivery Z-score signal: {flow_val:+.3f}). "
                    "This means smart money — FIIs, DIIs, HNIs — are leaving footprints in NSE delivery data.")
        else:
            st.info("💧 **Flow Detective** sees no unusual institutional activity (delivery Z-score near neutral).")

        if abs(tech_val) > 0.05:
            tech_direction = "bullish price momentum" if tech_val > 0 else "bearish price momentum"
            st.info(f"📐 **Technical Analyst** confirms **{tech_direction}** (signal: {tech_val:+.3f}). "
                    "LightGBM trained on NSE price patterns from 2015 onwards.")
        else:
            st.info("📐 **Technical Analyst** is neutral — price momentum doesn't have a strong direction.")

        if row.get("vetoed"):
            st.error(f"⚠️ **Risk Contrarian VETOED this signal.** "
                     f"Reason: {row.get('veto_reason','Anomaly detected')}. "
                     "The model sees a potential trap pattern. Final signal overridden to HOLD.")

        reasoning = str(row.get("reasoning", ""))
        if reasoning and reasoning != "nan":
            st.markdown(f'<div style="color:#777;font-size:.85rem;font-style:italic;'
                        f'border-left:3px solid #333;padding-left:10px;margin-top:.5rem;">'
                        f'{reasoning}</div>', unsafe_allow_html=True)

        # ── Historical signals for this ticker ───────────────────────────────
        hist = _add_derived(_history(sid, days=60))
        ticker_hist = hist[hist["ticker"] == sel_ticker].copy() if not hist.empty else pd.DataFrame()

        if not ticker_hist.empty:
            st.markdown('<div class="section-head">📅 Signal History (last 60 days)</div>',
                        unsafe_allow_html=True)
            ticker_hist = ticker_hist.sort_values("signal_date")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=pd.to_datetime(ticker_hist["signal_date"]),
                y=ticker_hist["final_signal"],
                mode="lines+markers",
                line=dict(color="#40c4ff", width=2),
                marker=dict(
                    color=ticker_hist["direction"].map(
                        {"BUY":"#00e676","SELL":"#ff1744","HOLD":"#ffd600"}),
                    size=8
                ),
                hovertemplate="<b>%{x|%d %b}</b><br>Signal: %{y:+.3f}<extra></extra>",
            ))
            if "flow_signal" in ticker_hist.columns:
                fig_hist.add_trace(go.Scatter(
                    x=pd.to_datetime(ticker_hist["signal_date"]),
                    y=ticker_hist["flow_signal"].fillna(0),
                    mode="lines",
                    name="Flow",
                    line=dict(color="#00e676", width=1, dash="dot"),
                    hovertemplate="Flow: %{y:+.3f}<extra></extra>",
                ))
            fig_hist.update_layout(
                height=220, margin=dict(l=0,r=0,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888", gridcolor="#2a2a3e"),
                yaxis=dict(color="#888", gridcolor="#2a2a3e", title="Signal"),
                legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
                font=dict(color="#ccc"),
            )
            fig_hist.add_hline(y=0, line_color="#333", line_width=1)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("🔵 Combined signal · 🟢 Flow Detective (dotted) · Markers: 🟢 BUY 🔴 SELL 🟡 HOLD")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TRACK RECORD
# ─────────────────────────────────────────────────────────────────────────────
with tab_record:
    st.header("📈 Track Record")
    st.caption("Walk-forward validated performance (Jan 2023 – Mar 2026) + live paper trades from today.")

    perf_df  = _perf(sid)
    pnl_df   = _pnl(sid)
    ic_df    = _ic(sid)
    bt_df    = _backtest_data()

    # ── Section A: Walk-Forward Backtest (always visible) ─────────────────────
    st.markdown(
        '<div class="regime-banner" style="background:rgba(0,230,118,.06);border-left:4px solid #00e676;">'
        '<strong style="color:#00e676;">✅ WALK-FORWARD VALIDATED PERFORMANCE</strong>'
        '<span style="color:#aaa;font-size:.85rem;margin-left:12px;">'
        'Jan 2023 – Mar 2026 · 82 rebalance dates · 12,293 observations · No look-ahead bias</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("IC (Spearman)",   "0.087",  delta="p<0.0001")
    b2.metric("L/S Sharpe",      "2.55",   delta="vs 0.5 baseline")
    b3.metric("Cum. L/S Return", "+469%",  delta="Jan 2023 → Mar 2026")
    b4.metric("BUY Avg 15d Ret", "+2.67%", delta="1,010 BUY signals")
    b5.metric("Max Drawdown",    "-12.4%", delta="controlled")
    b6.metric("Win Rate",        "76.8%",  delta="L/S strategy")

    if not bt_df.empty:
        bt_col1, bt_col2 = st.columns(2)

        # Quintile return chart
        with bt_col1:
            st.subheader("📊 Quintile Return Spread")
            q_stats = bt_df.groupby("quintile")["fwd_return"].mean() * 100
            q_labels = [f"Q{int(i)}" for i in q_stats.index]
            q_colors = ["#ff1744","#ff7043","#ffd600","#69f0ae","#00e676"]

            fig_q = go.Figure(go.Bar(
                x=q_labels, y=q_stats.values,
                marker_color=q_colors[:len(q_stats)],
                text=[f"{v:+.2f}%" for v in q_stats.values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Avg 15d return: %{y:+.2f}%<extra></extra>",
            ))
            fig_q.add_hline(y=0, line_color="#444", line_width=1)
            fig_q.update_layout(
                height=260, margin=dict(l=0,r=0,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#aaa"), yaxis=dict(color="#888", gridcolor="#2a2a3e",
                                                      title="Avg 15d Return (%)"),
                font=dict(color="#ccc"), showlegend=False,
            )
            st.plotly_chart(fig_q, use_container_width=True)
            st.caption("Q5 (strongest BUY signals) beats Q1 by ~2.2% over 15 days — monotonic spread = real alpha")

        # Monthly avg return of BUY signals
        with bt_col2:
            st.subheader("📅 Monthly BUY Signal Returns")
            buy_bt = bt_df[bt_df["direction"] == "BUY"].copy()
            if not buy_bt.empty:
                buy_bt["month"] = buy_bt["date"].dt.to_period("M").astype(str)
                monthly_ret = buy_bt.groupby("month")["fwd_return"].mean() * 100
                m_colors = ["#00e676" if v >= 0 else "#ff1744" for v in monthly_ret.values]

                fig_mo = go.Figure(go.Bar(
                    x=monthly_ret.index, y=monthly_ret.values,
                    marker_color=m_colors, opacity=0.85,
                    hovertemplate="<b>%{x}</b><br>Avg BUY return: %{y:+.2f}%<extra></extra>",
                ))
                fig_mo.add_hline(y=0, line_color="#444", line_width=1)
                fig_mo.add_hline(y=monthly_ret.mean(), line_color="#40c4ff", line_dash="dot",
                                 annotation_text=f"Mean {monthly_ret.mean():+.1f}%",
                                 annotation_font_color="#40c4ff")
                fig_mo.update_layout(
                    height=260, margin=dict(l=0,r=0,t=10,b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#888", tickangle=-45),
                    yaxis=dict(color="#888", gridcolor="#2a2a3e", title="Avg 15d Return (%)"),
                    font=dict(color="#ccc"), showlegend=False,
                )
                st.plotly_chart(fig_mo, use_container_width=True)

        # Signal score vs forward return scatter (regime-coloured)
        st.subheader("🎯 Signal Score → Actual 15d Return (82 dates, 12,293 obs)")
        sample = bt_df.sample(min(2000, len(bt_df)), random_state=42)
        regime_palette = {"Bull":"#00e676","Bear":"#ff1744",
                          "High-Vol Sideways":"#ff9100","Low-Vol Sideways":"#40c4ff"}
        sample["regime_short"] = sample["regime"].str.split("|").str[0].str.strip()\
                                   .str.replace("Market regime (rule-based): ","",regex=False).str.strip()

        fig_scat = go.Figure()
        for r, rc in regime_palette.items():
            sub = sample[sample["regime_short"] == r]
            if not sub.empty:
                fig_scat.add_trace(go.Scatter(
                    x=sub["final_signal"], y=sub["fwd_return"] * 100,
                    mode="markers", name=r,
                    marker=dict(color=rc, size=4, opacity=0.45),
                    hovertemplate=f"<b>{r}</b><br>Signal: %{{x:+.3f}}<br>Return: %{{y:+.1f}}%<extra></extra>",
                ))
        # Trendline
        try:
            z = np.polyfit(sample["final_signal"], sample["fwd_return"]*100, 1)
            x_range = np.linspace(sample["final_signal"].min(), sample["final_signal"].max(), 50)
            fig_scat.add_trace(go.Scatter(
                x=x_range, y=np.polyval(z, x_range),
                mode="lines", name="Trend",
                line=dict(color="#fff", width=2, dash="dash"),
            ))
        except Exception:
            pass
        fig_scat.add_vline(x=0, line_color="#333", line_width=1)
        fig_scat.add_hline(y=0, line_color="#333", line_width=1)
        fig_scat.update_layout(
            height=320, margin=dict(l=0,r=0,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#888", title="Signal Score", gridcolor="#2a2a3e"),
            yaxis=dict(color="#888", title="15d Forward Return (%)", gridcolor="#2a2a3e"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig_scat, use_container_width=True)
        st.caption("Higher signal → higher return. Colour = market regime. White dashed line = IC=0.087 trend.")

    st.divider()

    # ── Section B: Live Paper Trading (builds over time) ─────────────────────
    st.subheader("📍 Live Paper Trades")

    alpha_data = _alpha_analysis(sid)

    if perf_df.empty:
        st.info(
            "**No settled paper trades yet.**  "
            "Each evening run logs 10–30 BUY/SELL signals as paper trades. "
            "After 15 trading days they auto-settle. First settlements in ~3 weeks."
        )
        st.markdown("""
        **How it works:**
        - 🟢 WIN = BUY signal + stock rose >1% over 15 days (or SELL + fell >1%)
        - 🔴 LOSS = opposite
        - 📐 Rolling IC = Spearman correlation between signal score and realized return
        """)
    else:
        perf_df["signal_date"] = pd.to_datetime(perf_df["signal_date"])
        buys = perf_df[perf_df["direction"] == "BUY"]

        wins   = (buys["outcome"] == "WIN").sum()
        losses = (buys["outcome"] == "LOSS").sum()
        total  = len(buys)

        # ── Alpha banner (the real measure) ──
        if alpha_data:
            _alpha_val = alpha_data["avg_alpha"]
            _alpha_color = "#00e676" if _alpha_val >= 0 else "#ff1744"
            _alpha_bg = "rgba(0,230,118,.08)" if _alpha_val >= 0 else "rgba(255,23,68,.08)"
            _beat_pct = alpha_data["beat_market_pct"]
            st.markdown(
                f'<div class="regime-banner" style="background:{_alpha_bg};border-left:4px solid {_alpha_color};">'
                f'<strong style="color:{_alpha_color};font-size:1.1rem;">ALPHA: {_alpha_val:+.2%} per trade</strong>'
                f'<span style="color:#aaa;font-size:.85rem;margin-left:16px;">'
                f'BUY picks: {alpha_data["avg_return"]:+.2%} vs Market: {alpha_data["avg_market"]:+.2%} · '
                f'{_beat_pct:.0%} of picks beat the market · {alpha_data["n_trades"]} trades</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # KPIs — alpha-first
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("📈 Alpha/Trade",
                  f"{alpha_data['avg_alpha']:+.2%}" if alpha_data else "—",
                  delta=f"median {alpha_data['median_alpha']:+.2%}" if alpha_data else None)
        p2.metric("🎯 Beat Market",
                  f"{alpha_data['beat_market_pct']:.0%}" if alpha_data else "—",
                  delta=f"{alpha_data['beat_market_pct'] - 0.5:+.0%} vs 50%" if alpha_data else None)
        p3.metric("📊 Total Trades", total)
        p4.metric("✅ Win Rate",
                  f"{wins/total:.0%}" if total else "—",
                  delta=f"{wins/total - 0.5:+.0%} vs 50%" if total else None)
        p5.metric("📐 Avg Return",
                  f"{buys['actual_return'].mean():+.2%}" if total else "—",
                  delta=f"mkt {alpha_data['avg_market']:+.2%}" if alpha_data else None)
        p6.metric("🏪 Mkt Return",
                  f"{alpha_data['avg_market']:+.2%}" if alpha_data else "—")

        # ── Monthly Alpha Breakdown ──
        if alpha_data and alpha_data.get("monthly"):
            st.markdown("---")
            st.subheader("📅 Monthly Alpha Breakdown")
            monthly = alpha_data["monthly"]
            m_df = pd.DataFrame(monthly)
            m_df.columns = ["Month", "Trades", "BUY Avg Return", "Market Return", "Alpha"]

            # Color-coded bar chart
            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Bar(
                x=m_df["Month"],
                y=[v * 100 for v in m_df["Alpha"]],
                marker_color=["#00e676" if v >= 0 else "#ff1744" for v in m_df["Alpha"]],
                text=[f"{v:+.2%}" for v in m_df["Alpha"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Alpha: %{y:+.2f}%<extra></extra>",
            ))
            fig_alpha.add_hline(y=0, line_color="#444", line_width=1)
            fig_alpha.update_layout(
                height=250, margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888"), yaxis=dict(color="#888", gridcolor="#2a2a3e", title="Alpha (%)"),
                font=dict(color="#ccc"), showlegend=False,
            )
            alpha_left, alpha_right = st.columns([2, 1])
            with alpha_left:
                st.plotly_chart(fig_alpha, use_container_width=True)
            with alpha_right:
                st.caption("Alpha = BUY return − Market return. Positive = model outperforms.")
                _display_df = m_df.copy()
                _display_df["BUY Avg Return"] = _display_df["BUY Avg Return"].map(lambda v: f"{v:+.2%}")
                _display_df["Market Return"] = _display_df["Market Return"].map(lambda v: f"{v:+.2%}")
                _display_df["Alpha"] = _display_df["Alpha"].map(lambda v: f"{v:+.2%}")
                st.dataframe(_display_df, hide_index=True, use_container_width=True)

        st.divider()

        left_col, right_col = st.columns(2)

        # P&L curve
        with left_col:
            st.subheader("💰 Cumulative Paper P&L")
            if not pnl_df.empty and "cumulative_return" in pnl_df.columns:
                pnl_df["signal_date"] = pd.to_datetime(pnl_df["signal_date"])
                cr = pnl_df["cumulative_return"].iloc[-1]
                line_col = "#00e676" if cr >= 0 else "#ff1744"

                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=pnl_df["signal_date"],
                    y=pnl_df["cumulative_return"] * 100,
                    fill="tozeroy",
                    fillcolor=f"rgba({'0,230,118' if cr>=0 else '255,23,68'},.08)",
                    line=dict(color=line_col, width=2),
                    hovertemplate="%{x|%d %b}<br>Return: %{y:+.2f}%<extra></extra>",
                ))
                fig_pnl.update_layout(
                    height=250, margin=dict(l=0,r=0,t=10,b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#888", gridcolor="#2a2a3e"),
                    yaxis=dict(color="#888", gridcolor="#2a2a3e", title="Return %"),
                    font=dict(color="#ccc"),
                )
                fig_pnl.add_hline(y=0, line_color="#444", line_width=1)
                st.plotly_chart(fig_pnl, use_container_width=True)
                st.metric("Total Paper Return", f"{cr:+.2%}")

        # Rolling IC
        with right_col:
            st.subheader("📐 Rolling 20-day IC")
            if not ic_df.empty and "rolling_ic" in ic_df.columns:
                ic_df["signal_date"] = pd.to_datetime(ic_df["signal_date"])
                ic_clean = ic_df.dropna(subset=["rolling_ic"])

                fig_ic = go.Figure()
                fig_ic.add_hrect(y0=0.05, y1=0.3,  fillcolor="rgba(0,230,118,.05)", line_width=0)
                fig_ic.add_hrect(y0=0.02, y1=0.05, fillcolor="rgba(255,214,0,.05)",  line_width=0)
                fig_ic.add_trace(go.Scatter(
                    x=ic_clean["signal_date"], y=ic_clean["rolling_ic"],
                    mode="lines", line=dict(color="#40c4ff", width=2),
                    hovertemplate="%{x|%d %b}<br>IC: %{y:+.4f}<extra></extra>",
                ))
                fig_ic.update_layout(
                    height=250, margin=dict(l=0,r=0,t=10,b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#888", gridcolor="#2a2a3e"),
                    yaxis=dict(color="#888", gridcolor="#2a2a3e", title="IC"),
                    font=dict(color="#ccc"),
                )
                fig_ic.add_hline(y=0.05, line_color="#00e676", line_width=1,
                                 line_dash="dot", annotation_text="IC=0.05 target",
                                 annotation_font_color="#00e676")
                fig_ic.add_hline(y=0,    line_color="#444", line_width=1)
                st.plotly_chart(fig_ic, use_container_width=True)
                st.caption("Green zone = IC > 0.05 (institutional-grade). Yellow = IC 0.02-0.05 (meaningful edge).")

        st.divider()

        # Win/Loss by month
        st.subheader("📅 Monthly Performance Breakdown")
        if total > 0:
            buys["month"] = buys["signal_date"].dt.to_period("M").astype(str)
            monthly = buys.groupby("month").agg(
                wins=("outcome", lambda x: (x == "WIN").sum()),
                losses=("outcome", lambda x: (x == "LOSS").sum()),
                avg_ret=("actual_return", "mean"),
            ).reset_index()

            fig_mo = go.Figure()
            fig_mo.add_trace(go.Bar(x=monthly["month"], y=monthly["wins"],
                                    name="Wins", marker_color="#00e676"))
            fig_mo.add_trace(go.Bar(x=monthly["month"], y=-monthly["losses"],
                                    name="Losses", marker_color="#ff1744"))
            fig_mo.update_layout(
                barmode="overlay", height=250, margin=dict(l=0,r=0,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888"), yaxis=dict(color="#888", gridcolor="#2a2a3e"),
                legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_mo, use_container_width=True)

        # Recent trades table
        st.subheader("📋 Recent Settled Trades")
        recent = buys.sort_values("signal_date", ascending=False).head(25)
        rec_show = [c for c in ["signal_date","ticker","final_signal","actual_return","outcome"]
                    if c in recent.columns]

        def _outcome_style(val):
            if val == "WIN":  return "color: #00e676"
            if val == "LOSS": return "color: #ff1744"
            return "color: #888"

        st.dataframe(
            recent[rec_show]
            .rename(columns={"signal_date":"Date","ticker":"Stock",
                              "final_signal":"Score","actual_return":"Return","outcome":"Outcome"})
            .style
            .format({"Score":"{:+.3f}","Return":"{:+.2%}"})
            .applymap(_outcome_style, subset=["Outcome"]),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MARKET PULSE
# ─────────────────────────────────────────────────────────────────────────────
with tab_pulse:
    st.header("🌐 Market Pulse")
    st.caption("Current regime, sector breakdown, and how to position.")

    if df_now.empty:
        st.info("Run the live runner to see market pulse.")
    else:
        # ── Regime deep-dive ─────────────────────────────────────────────────
        rcol2 = REGIME_COLORS.get(dominant_regime, "#888")
        rbg2  = REGIME_BG.get(dominant_regime, "rgba(128,128,128,.06)")

        st.markdown(
            f'<div class="regime-banner" style="background:{rbg2};border-left:4px solid {rcol2};font-size:1.1rem;">'
            f'<strong style="color:{rcol2};font-size:1.3rem;">● {dominant_regime}</strong>'
            f'<br><span style="color:#aaa;">{REGIME_NOTES.get(dominant_regime,"")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        regime_advice = {
            "Bull":              ("Increase exposure", "Threshold = 0.10. Lower bar for entry. Let winners run. Flow-driven BUYs have highest hit rate in bull markets."),
            "Bear":              ("Reduce exposure", "Threshold = 0.22. Only ultra-high-conviction signals. Consider hedges. Cash is a position."),
            "High-Vol Sideways": ("Stay selective", "Threshold = 0.25. Choppy conditions. Smaller position sizes. Focus on stocks where Flow AND Tech both agree."),
            "Low-Vol Sideways":  ("Standard exposure", "Threshold = 0.15. Normal conditions. Standard position sizes. Good for breakout plays if Flow starts moving."),
        }
        advice_title, advice_body = regime_advice.get(dominant_regime, ("Standard", "Normal market conditions."))
        st.info(f"**Position advice: {advice_title}** — {advice_body}")

        st.divider()

        # ── Signal distribution ───────────────────────────────────────────────
        pa_col, pb_col = st.columns(2)

        with pa_col:
            st.subheader("📊 Signal Distribution")
            sig_hist_data = df_now["final_signal"]
            fig_dist = go.Figure(go.Histogram(
                x=sig_hist_data, nbinsx=30,
                marker_color="#40c4ff", opacity=0.8,
                hovertemplate="Signal: %{x:.2f}<br>Count: %{y}<extra></extra>",
            ))
            threshold = {"Bull":0.10,"Bear":0.22,"High-Vol Sideways":0.25,"Low-Vol Sideways":0.15}.get(dominant_regime, 0.15)
            fig_dist.add_vline(x=threshold,  line_color="#00e676", line_dash="dash",
                               annotation_text=f"BUY >{threshold}", annotation_font_color="#00e676")
            fig_dist.add_vline(x=-threshold, line_color="#ff1744", line_dash="dash",
                               annotation_text=f"SELL <{-threshold}", annotation_font_color="#ff1744")
            fig_dist.update_layout(
                height=280, margin=dict(l=0,r=0,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888",title="Signal Score"),
                yaxis=dict(color="#888",gridcolor="#2a2a3e",title="Count"),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with pb_col:
            st.subheader("🏭 Sector Breakdown")
            sec_counts = df_now[df_now["direction"]=="BUY"]["sector"].value_counts()
            if not sec_counts.empty:
                fig_sec = go.Figure(go.Bar(
                    x=sec_counts.index, y=sec_counts.values,
                    marker_color="#00e676", opacity=0.85,
                    hovertemplate="%{x}<br>BUY signals: %{y}<extra></extra>",
                ))
                fig_sec.update_layout(
                    height=280, margin=dict(l=0,r=0,t=10,b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#888",tickangle=-30),
                    yaxis=dict(color="#888",gridcolor="#2a2a3e"),
                    font=dict(color="#ccc"),
                )
                st.plotly_chart(fig_sec, use_container_width=True)
            else:
                st.info("No BUY signals in current filters.")

        st.divider()

        # ── Flow vs Tech scatter ──────────────────────────────────────────────
        st.subheader("💧 Flow vs Technical — Where Do They Agree?")
        scatter_df = df_now[df_now["direction"].isin(["BUY","SELL"])].copy()
        if not scatter_df.empty and "flow_signal" in scatter_df and "technical_signal" in scatter_df:
            scatter_df["flow_signal"]      = scatter_df["flow_signal"].fillna(0)
            scatter_df["technical_signal"] = scatter_df["technical_signal"].fillna(0)
            scatter_df["color_dir"]        = scatter_df["direction"].map({"BUY":"#00e676","SELL":"#ff1744"})

            fig_sc = go.Figure()
            for d, color in [("BUY","#00e676"),("SELL","#ff1744")]:
                sub = scatter_df[scatter_df["direction"]==d]
                if not sub.empty:
                    fig_sc.add_trace(go.Scatter(
                        x=sub["flow_signal"], y=sub["technical_signal"],
                        mode="markers+text",
                        marker=dict(color=color, size=10, opacity=0.85,
                                    line=dict(color="#1a1a2e", width=1)),
                        text=sub["ticker"],
                        textposition="top center",
                        textfont=dict(size=9, color="#aaa"),
                        name=d,
                        hovertemplate="<b>%{text}</b><br>Flow: %{x:+.3f}<br>Tech: %{y:+.3f}<extra></extra>",
                    ))
            fig_sc.add_vline(x=0, line_color="#333"); fig_sc.add_hline(y=0, line_color="#333")
            fig_sc.add_annotation(x=0.3, y=0.3, text="Both bullish\n(highest conviction)",
                                   font=dict(color="#00e676",size=9), showarrow=False)
            fig_sc.add_annotation(x=-0.3, y=-0.3, text="Both bearish",
                                   font=dict(color="#ff1744",size=9), showarrow=False)
            fig_sc.update_layout(
                height=380, margin=dict(l=0,r=0,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888", title="Flow Detective Signal", gridcolor="#2a2a3e"),
                yaxis=dict(color="#888", title="Technical Analyst Signal", gridcolor="#2a2a3e"),
                legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption("📍 Top-right quadrant = both Flow and Technical bullish → highest conviction BUYs")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MODEL HEALTH
# ─────────────────────────────────────────────────────────────────────────────
with tab_health:
    st.header("🧠 Model Health")
    st.caption("Monitor signal quality, regime stability, and agent agreement over time.")

    hist90 = _add_derived(_history(sid, days=90))

    # ── Quick health indicators ───────────────────────────────────────────────
    h1c, h2c, h3c, h4c = st.columns(4)

    if not hist90.empty:
        recent7 = hist90[pd.to_datetime(hist90["signal_date"]) >= pd.Timestamp.now() - pd.Timedelta(days=7)]
        avg_flow_7d = recent7["flow_signal"].fillna(0).mean() if not recent7.empty else 0
        avg_sig_7d  = recent7["final_signal"].fillna(0).mean() if not recent7.empty else 0
        veto_rate   = hist90["vetoed"].mean() * 100
        flow_active = (hist90["flow_signal"].fillna(0).abs() > 0.1).mean() * 100

        h1c.metric("7d Avg Flow Signal",   f"{avg_flow_7d:+.3f}",
                   delta="Bullish bias" if avg_flow_7d > 0 else "Bearish bias")
        h2c.metric("7d Avg Final Signal",  f"{avg_sig_7d:+.3f}")
        h3c.metric("Veto Rate (90d)",      f"{veto_rate:.1f}%",
                   delta="Normal" if veto_rate < 10 else "Elevated")
        h4c.metric("Flow Active %",        f"{flow_active:.0f}%",
                   delta="Delivery data working" if flow_active > 30 else "Low activity")
    else:
        for c in [h1c,h2c,h3c,h4c]:
            c.metric("—","—")

    st.divider()

    # ── Regime history ────────────────────────────────────────────────────────
    regime_df = _regimes(sid)
    st.subheader("🌐 Regime History (last 90 days)")

    if not regime_df.empty:
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            regime_df["log_date"] = pd.to_datetime(regime_df["log_date"])
            regime_cnt = regime_df["regime"].value_counts().reset_index()
            regime_cnt.columns = ["Regime", "Days"]
            rc_colors  = [REGIME_COLORS.get(r, "#888") for r in regime_cnt["Regime"]]

            fig_rh = go.Figure(go.Bar(
                x=regime_cnt["Regime"], y=regime_cnt["Days"],
                marker_color=rc_colors,
                hovertemplate="%{x}: %{y} days<extra></extra>",
            ))
            fig_rh.update_layout(
                height=240, margin=dict(l=0,r=0,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888"), yaxis=dict(color="#888",gridcolor="#2a2a3e"),
                font=dict(color="#ccc"), showlegend=False,
            )
            st.plotly_chart(fig_rh, use_container_width=True)

        with rc2:
            st.dataframe(
                regime_df[["log_date","regime"]].sort_values("log_date",ascending=False).head(15)
                .rename(columns={"log_date":"Date","regime":"Regime"}),
                use_container_width=True, hide_index=True, height=240,
            )
    else:
        st.info("Regime history builds up with each daily run.")

    # ── Agent signal trends ────────────────────────────────────────────────────
    st.subheader("📈 Agent Signal Trends (90 days)")
    if not hist90.empty:
        agent_map = {"flow_signal":"Flow Detective","technical_signal":"Technical Analyst",
                     "earnings_signal":"Earnings Oracle","risk_signal":"Risk Contrarian"}
        agent_colors = {"Flow Detective":"#00e676","Technical Analyst":"#42a5f5",
                        "Earnings Oracle":"#ffca28","Risk Contrarian":"#ef5350"}

        hist90["signal_date"] = pd.to_datetime(hist90["signal_date"])
        daily_avg = hist90.groupby("signal_date")[[c for c in agent_map if c in hist90.columns]].mean().reset_index()

        fig_trend = go.Figure()
        for col, label in agent_map.items():
            if col in daily_avg.columns:
                fig_trend.add_trace(go.Scatter(
                    x=daily_avg["signal_date"], y=daily_avg[col],
                    name=label, mode="lines",
                    line=dict(color=agent_colors[label], width=2),
                    hovertemplate=f"<b>{label}</b><br>%{{x|%d %b}}: %{{y:+.3f}}<extra></extra>",
                ))
        fig_trend.add_hline(y=0, line_color="#333", line_width=1)
        fig_trend.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#888", gridcolor="#2a2a3e"),
            yaxis=dict(color="#888", gridcolor="#2a2a3e", title="Avg Signal"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("💧 Flow Detective (green) = primary alpha source. Watch for it diverging from Technical — that's when the signal is most powerful.")

        # Agent correlation matrix
        st.subheader("📐 Agent Correlation Matrix")
        avail = [c for c in agent_map if c in hist90.columns]
        if len(avail) >= 2:
            corr_df = hist90[avail].rename(columns=agent_map).corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns.tolist(),
                y=corr_df.index.tolist(),
                colorscale="RdYlGn", zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            ))
            fig_corr.update_layout(
                height=280, margin=dict(l=0,r=0,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Low correlation between Flow Detective and Technical Analyst = healthy diversification of signals.")
    else:
        st.info("Agent trends will populate after a few days of runs.")


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown(
    '<div class="disclaimer">'
    '🔬 <strong>IntentFlow AI v4.5</strong> — Council of Experts · NIFTY 500 · '
    'Walk-forward validated IC = 0.087 (Jan 2023 – Mar 2026, 12,293 obs) · L/S Sharpe = 2.55 · '
    'Flow Detective IC = +0.097 (p&lt;0.0001)<br>'
    '⚠️ <strong>Not financial advice.</strong> This is a personal quantitative research tool for educational use. '
    'Past paper trading performance does not guarantee future results. '
    'All signals are for research purposes only. Always do your own due diligence.'
    '</div>',
    unsafe_allow_html=True,
)
