"""Streamlit dashboard for IntentFlow AI - Trader Edition.

A trader-friendly interface for viewing stock signals with clear Buy/Avoid recommendations,
signal strength indicators, and position sizing guidance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IntentFlow AI - Trader Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for trader-friendly styling
st.markdown("""
<style>
    .signal-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .strong-buy { 
        background: linear-gradient(90deg, rgba(0,200,83,0.15) 0%, rgba(0,200,83,0.05) 100%);
        border-left-color: #00c853;
    }
    .buy { 
        background: linear-gradient(90deg, rgba(76,175,80,0.15) 0%, rgba(76,175,80,0.05) 100%);
        border-left-color: #4caf50;
    }
    .hold { 
        background: linear-gradient(90deg, rgba(255,193,7,0.15) 0%, rgba(255,193,7,0.05) 100%);
        border-left-color: #ffc107;
    }
    .avoid { 
        background: linear-gradient(90deg, rgba(244,67,54,0.15) 0%, rgba(244,67,54,0.05) 100%);
        border-left-color: #f44336;
    }
    .metric-big {
        font-size: 2rem;
        font-weight: bold;
    }
    .stDataFrame { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


def get_signal(proba: float, p80: float = 0.41, p60: float = 0.38, p40: float = 0.36) -> Tuple[str, str, str]:
    """Translate probability to trader-friendly signal using percentile-based thresholds.
    
    The V2 ensemble model outputs conservative probabilities (35% mean, 52% max).
    Thresholds are based on percentiles:
    - Strong Buy: top 20% (>= p80)
    - Buy: 60-80% percentile (>= p60)
    - Hold: 40-60% percentile (>= p40)
    - Avoid: bottom 40% (< p40)
    
    Returns: (signal_text, conviction_level, css_class)
    """
    if proba >= p80:
        return "🟢 STRONG BUY", "High", "strong-buy"
    elif proba >= p60:
        return "🟢 BUY", "Medium", "buy"
    elif proba >= p40:
        return "🟡 HOLD", "Low", "hold"
    else:
        return "🔴 AVOID", "Low", "avoid"



def get_position_size(conviction: str, risk_pct: float = 2.0) -> str:
    """Suggest position size based on conviction level."""
    multipliers = {"High": 1.5, "Medium": 1.0, "Low": 0.5}
    suggested = risk_pct * multipliers.get(conviction, 0.5)
    return f"{suggested:.1f}% of portfolio"


@st.cache_data(show_spinner=False)
def load_predictions(exp_dir: Path) -> pd.DataFrame:
    """Load predictions data."""
    preds_path = exp_dir / "preds.csv"
    if not preds_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(preds_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "proba"], ascending=[False, False])


@st.cache_data(show_spinner=False)
def load_metrics(exp_dir: Path) -> Dict:
    """Load experiment metrics."""
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


@st.cache_data(show_spinner=False)
def load_wfo_summary(exp_dir: Path) -> pd.DataFrame:
    """Load walk-forward optimization summary."""
    wfo_path = exp_dir / "walk_forward_summary.csv"
    if wfo_path.exists():
        return pd.read_csv(wfo_path)
    return pd.DataFrame()


def get_sector_map() -> Dict[str, str]:
    """Return sector mapping for stocks (can be extended)."""
    # Basic sector mapping - will be enriched from data
    return {
        "TATASTEEL": "Materials", "HINDALCO": "Materials", "ULTRACEMCO": "Materials",
        "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "TECHM": "IT", "HCLTECH": "IT",
        "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
        "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials", "KOTAKBANK": "Financials",
        "SUNPHARMA": "Healthcare", "DRREDDY": "Healthcare", "CIPLA": "Healthcare",
        "TITAN": "Consumer", "HINDUNILVR": "Consumer", "ITC": "Consumer",
        "ADANIENT": "Industrials", "LT": "Industrials", "VOLTAS": "Industrials",
    }


# === MAIN APP ===
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Experiment selector
    exp_dirs = [d.name for d in EXPERIMENTS_DIR.iterdir() 
                if d.is_dir() and not d.name.startswith(".")]
    default_exp = "v_universe_full" if "v_universe_full" in exp_dirs else exp_dirs[0] if exp_dirs else ""
    selected_exp = st.selectbox("📁 Experiment", exp_dirs, 
                                 index=exp_dirs.index(default_exp) if default_exp in exp_dirs else 0)
    
    st.divider()
    
    # Filters
    st.subheader("🎯 Filters")
    min_conviction = st.select_slider(
        "Min Conviction", 
        options=["Low", "Medium", "High"],
        value="Low"
    )
    
    show_signals = st.multiselect(
        "Signal Types",
        ["🟢 STRONG BUY", "🟢 BUY", "🟡 HOLD", "🔴 AVOID"],
        default=["🟢 STRONG BUY", "🟢 BUY"]
    )
    
    st.divider()
    
    # Position sizing settings
    st.subheader("💰 Position Sizing")
    risk_per_trade = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0, 0.5)

# Load data
EXP_DIR = EXPERIMENTS_DIR / selected_exp
preds = load_predictions(EXP_DIR)
metrics = load_metrics(EXP_DIR)
wfo_summary = load_wfo_summary(EXP_DIR)

# Header
st.title("📊 IntentFlow AI - Trader Dashboard")
st.caption("Actionable trading signals for NIFTY 200 universe | Updated daily")

if preds.empty:
    st.error("❌ No predictions found. Run the model training first.")
    st.stop()

# Get latest date and prepare data
latest_date = preds["date"].max()
latest_preds = preds[preds["date"] == latest_date].copy()

# Calculate percentile-based thresholds for signal generation
# V2 model gives conservative probabilities, so we use relative ranking
p80 = latest_preds["proba"].quantile(0.80)  # Top 20% = Strong Buy
p60 = latest_preds["proba"].quantile(0.60)  # 60-80% = Buy  
p40 = latest_preds["proba"].quantile(0.40)  # 40-60% = Hold, <40% = Avoid

# Add signal columns using dynamic thresholds
latest_preds["signal_info"] = latest_preds["proba"].apply(lambda x: get_signal(x, p80, p60, p40))
latest_preds["signal"] = latest_preds["signal_info"].apply(lambda x: x[0])
latest_preds["conviction"] = latest_preds["signal_info"].apply(lambda x: x[1])
latest_preds["css_class"] = latest_preds["signal_info"].apply(lambda x: x[2])

# Add sector
sector_map = get_sector_map()
latest_preds["sector"] = latest_preds["ticker"].map(sector_map).fillna("Other")

# Add expected return
latest_preds["exp_return"] = latest_preds["excess_fwd"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "N/A")

# === TABS ===
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Today's Signals", 
    "🔍 Stock Screener", 
    "🎯 Watchlist Builder",
    "📈 Model Health"
])

# === TAB 1: TODAY'S SIGNALS ===
with tab1:
    st.header(f"📊 Today's Signals - {latest_date.strftime('%d %b %Y')}")
    
    # Headline metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    strong_buys = len(latest_preds[latest_preds["signal"] == "🟢 STRONG BUY"])
    buys = len(latest_preds[latest_preds["signal"] == "🟢 BUY"])
    holds = len(latest_preds[latest_preds["signal"] == "🟡 HOLD"])
    avoids = len(latest_preds[latest_preds["signal"] == "🔴 AVOID"])
    
    col1.metric("🟢 Strong Buys", strong_buys)
    col2.metric("🟢 Buys", buys)
    col3.metric("🟡 Holds", holds)
    col4.metric("🔴 Avoids", avoids)
    col5.metric("Total Signals", len(latest_preds))
    
    st.divider()
    
    # Filter by selected signals
    conv_order = {"Low": 0, "Medium": 1, "High": 2}
    min_conv_value = conv_order[min_conviction]
    
    filtered = latest_preds[
        (latest_preds["signal"].isin(show_signals)) &
        (latest_preds["conviction"].map(conv_order) >= min_conv_value)
    ].sort_values("proba", ascending=False)
    
    if filtered.empty:
        st.info("No signals match your filters. Try broadening your criteria.")
    else:
        # Top 10 grid view
        st.subheader(f"🏆 Top {min(10, len(filtered))} Picks")
        
        top_10 = filtered.head(10)
        cols = st.columns(2)
        
        for idx, (_, row) in enumerate(top_10.iterrows()):
            with cols[idx % 2]:
                css_class = row["css_class"]
                st.markdown(f"""
                <div class="signal-card {css_class}">
                    <h3 style="margin:0;">{row['ticker']}</h3>
                    <p style="margin:0; color:#666;">{row['sector']}</p>
                    <p style="margin:0.5rem 0; font-size:1.2rem;">{row['signal']}</p>
                    <p style="margin:0;">
                        <strong>Conviction:</strong> {row['conviction']} | 
                        <strong>Signal Score:</strong> {row['proba']:.1%}
                    </p>
                    <p style="margin:0; font-size:0.9rem; color:#888;">
                        Suggested Position: {get_position_size(row['conviction'], risk_per_trade)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Full list
        st.subheader("📋 All Filtered Signals")
        display_cols = ["ticker", "sector", "signal", "conviction", "proba"]
        display_df = filtered[display_cols].rename(columns={
            "ticker": "Stock",
            "sector": "Sector", 
            "signal": "Signal",
            "conviction": "Conviction",
            "proba": "Signal Strength"
        })
        
        st.dataframe(
            display_df.style.format({"Signal Strength": "{:.1%}"})
                     .background_gradient(subset=["Signal Strength"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True
        )

# === TAB 2: STOCK SCREENER ===
with tab2:
    st.header("🔍 Stock Screener")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sectors = ["All"] + sorted(latest_preds["sector"].unique().tolist())
        selected_sector = st.selectbox("Sector", sectors)
    
    with col2:
        min_prob = st.slider("Min Signal Strength", 0.0, 1.0, 0.40, 0.05)
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Signal Strength", "Stock Name", "Sector"])
    
    # Apply filters
    screened = latest_preds.copy()
    if selected_sector != "All":
        screened = screened[screened["sector"] == selected_sector]
    screened = screened[screened["proba"] >= min_prob]
    
    # Sort
    sort_map = {"Signal Strength": "proba", "Stock Name": "ticker", "Sector": "sector"}
    ascending = sort_by != "Signal Strength"
    screened = screened.sort_values(sort_map[sort_by], ascending=ascending)
    
    st.metric("Matching Stocks", len(screened))
    
    # Display table
    screen_df = screened[["ticker", "sector", "signal", "conviction", "proba", "exp_return"]].rename(columns={
        "ticker": "Stock",
        "sector": "Sector",
        "signal": "Signal", 
        "conviction": "Conviction",
        "proba": "Signal Strength",
        "exp_return": "Exp. Return (10d)"
    })
    
    st.dataframe(
        screen_df.style.format({"Signal Strength": "{:.1%}"})
                 .background_gradient(subset=["Signal Strength"], cmap="RdYlGn", vmin=0.3, vmax=0.7),
        use_container_width=True,
        hide_index=True,
        height=500
    )
    
    # Export button
    csv = screen_df.to_csv(index=False)
    st.download_button(
        "📥 Export to CSV",
        csv,
        f"signals_{latest_date.strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )

# === TAB 3: WATCHLIST BUILDER ===
with tab3:
    st.header("🎯 Watchlist Builder")
    
    st.info("Select stocks to build your watchlist and see portfolio-level insights.")
    
    # Multi-select stocks
    available_stocks = latest_preds.sort_values("proba", ascending=False)["ticker"].tolist()
    selected_stocks = st.multiselect(
        "Add stocks to watchlist",
        available_stocks,
        default=available_stocks[:5]  # Default to top 5
    )
    
    if selected_stocks:
        watchlist = latest_preds[latest_preds["ticker"].isin(selected_stocks)].copy()
        
        # Portfolio summary
        col1, col2, col3 = st.columns(3)
        
        avg_conviction = watchlist["proba"].mean()
        high_conv_count = len(watchlist[watchlist["conviction"] == "High"])
        sector_count = watchlist["sector"].nunique()
        
        col1.metric("Avg Signal Strength", f"{avg_conviction:.1%}")
        col2.metric("High Conviction Picks", high_conv_count)
        col3.metric("Sectors Covered", sector_count)
        
        # Concentration warnings
        sector_weights = watchlist["sector"].value_counts(normalize=True)
        max_sector_weight = sector_weights.max()
        
        if max_sector_weight > 0.4:
            st.warning(f"⚠️ **Concentration Risk**: {sector_weights.idxmax()} makes up {max_sector_weight:.0%} of your watchlist")
        
        if len(selected_stocks) < 5:
            st.warning("⚠️ **Diversification**: Consider adding more stocks for better diversification")
        
        st.divider()
        
        # Watchlist table with position sizing
        st.subheader("📋 Your Watchlist")
        
        watchlist["position_size"] = watchlist["conviction"].apply(
            lambda c: get_position_size(c, risk_per_trade)
        )
        
        watch_df = watchlist[["ticker", "sector", "signal", "conviction", "proba", "position_size"]].rename(columns={
            "ticker": "Stock",
            "sector": "Sector",
            "signal": "Signal",
            "conviction": "Conviction",
            "proba": "Signal Strength",
            "position_size": "Suggested Position"
        })
        
        st.dataframe(
            watch_df.style.format({"Signal Strength": "{:.1%}"}),
            use_container_width=True,
            hide_index=True
        )
        
        # Sector breakdown
        st.subheader("📊 Sector Breakdown")
        st.bar_chart(watchlist["sector"].value_counts())
    else:
        st.info("👆 Select stocks above to build your watchlist")

# === TAB 4: MODEL HEALTH ===
with tab4:
    st.header("📈 Model Health")
    
    # Key metrics in trader-friendly language
    col1, col2, col3, col4 = st.columns(4)
    
    if "wfo_test" in metrics:
        wfo = metrics["wfo_test"]
        
        # Win rate from decile analysis
        decile_stats = wfo.get("decile_stats", [])
        if decile_stats:
            top_decile = decile_stats[-1]  # Top 10%
            bottom_decile = decile_stats[0]  # Bottom 10%
            
            col1.metric(
                "Top 10% Avg Return",
                f"{top_decile['mean_return']*100:+.1f}%",
                help="Average return of stocks in the top 10% signal strength"
            )
            col2.metric(
                "Bottom 10% Avg Return",
                f"{bottom_decile['mean_return']*100:+.1f}%",
                help="Average return of stocks in the bottom 10% signal strength"
            )
            col3.metric(
                "Edge (Top vs Bottom)",
                f"{(top_decile['mean_return'] - bottom_decile['mean_return'])*100:+.1f}%",
                help="The difference between top and bottom picks"
            )
        
        # Precision
        precision = wfo.get("precision_by_day_at_10", 0)
        col4.metric(
            "Daily Hit Rate (Top 10)",
            f"{precision:.0%}",
            help="% of days where top 10 picks beat the market"
        )
    
    st.divider()
    
    # Walk-forward performance chart
    st.subheader("📊 Historical Performance")
    
    if not wfo_summary.empty:
        wfo_summary["test_end"] = pd.to_datetime(wfo_summary["test_end"])
        
        # Rolling IC chart (simplified as "Model Accuracy")
        chart_data = wfo_summary[["test_end", "test_rank_ic"]].copy()
        chart_data.columns = ["Date", "Model Accuracy"]
        chart_data = chart_data.set_index("Date")
        
        st.line_chart(chart_data, use_container_width=True)
        st.caption("Higher values = better stock ranking accuracy. Values above 0.05 are considered good.")
        
        # Performance summary
        avg_ic = wfo_summary["test_rank_ic"].mean()
        positive_folds = (wfo_summary["test_rank_ic"] > 0).mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Avg Model Accuracy", f"{avg_ic:.3f}")
        col2.metric("% Positive Periods", f"{positive_folds:.0%}")
        
        # Recent performance
        st.subheader("📅 Recent Performance (Last 6 Periods)")
        recent = wfo_summary.tail(6)[["test_end", "test_rank_ic", "test_roc_auc"]].copy()
        recent.columns = ["Period End", "Accuracy", "AUC Score"]
        recent["Period End"] = pd.to_datetime(recent["Period End"]).dt.strftime("%b %Y")
        
        st.dataframe(
            recent.style.format({"Accuracy": "{:.3f}", "AUC Score": "{:.3f}"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Walk-forward summary not available. Run WFO training to see historical performance.")

# Footer
st.divider()
st.caption("""
**IntentFlow AI** | Systematic trading signals for NIFTY 200 universe  
⚠️ *Disclaimer: This is for educational purposes. Past performance does not guarantee future results. Always do your own research.*
""")
