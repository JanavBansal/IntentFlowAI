import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="IntentFlow AI Trader", layout="wide")

@st.cache_data
def load_data(experiment_name="v_universe_sanity"):
    exp_dir = Path("experiments") / experiment_name
    
    # Load predictions
    preds_path = exp_dir / "preds.csv"
    if not preds_path.exists():
        return None, None, None
    
    preds = pd.read_csv(preds_path)
    preds["date"] = pd.to_datetime(preds["date"])
    
    # Load metrics
    metrics_path = exp_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
            
    # Load feature importance
    fi_path = exp_dir / "feature_importance.csv"
    fi = pd.DataFrame()
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        
    return preds, metrics, fi

def main():
    st.title("IntentFlow AI: Trader Dashboard")
    
    preds, metrics, fi = load_data()
    
    if preds is None:
        st.error("No experiment data found. Run WFO training first.")
        return
        
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Experiment Selector
    exp_dir = Path("experiments")
    if exp_dir.exists():
        experiments = [d.name for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    else:
        experiments = []
    
    default_exp = "v_universe_sanity" if "v_universe_sanity" in experiments else (experiments[0] if experiments else "")
    selected_exp = st.sidebar.selectbox("Select Experiment", experiments, index=experiments.index(default_exp) if default_exp in experiments else 0)
    
    min_proba = st.sidebar.slider("Min Probability", 0.0, 1.0, 0.5)
    
    if not selected_exp:
        st.error("No experiments found in experiments/ directory.")
        return

    preds, metrics, fi = load_data(selected_exp)
    
    if preds is None:
        st.error(f"No data found for experiment '{selected_exp}'. Run training first.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Top Picks", "Model Health", "Alpha Drivers"])

    with tab1:
        st.header("Top Picks for Next Trading Day")
        latest_date = preds["date"].max()
        st.info(f"Latest Signal Date: {latest_date.date()}")
        
        latest_preds = preds[preds["date"] == latest_date].copy()
        latest_preds = latest_preds[latest_preds["proba"] >= min_proba]
        latest_preds = latest_preds.sort_values("proba", ascending=False).head(20)
        
        st.dataframe(
            latest_preds[["ticker", "proba", "excess_fwd"]].style.format({
                "proba": "{:.1%}",
                "excess_fwd": "{:.2%}"
            })
        )
        
    with tab2:
        st.header("Model Health (Walk-Forward)")
        
        # Cumulative Return
        # Assuming equal weight on top decile or similar
        # For now, just show IC over time if available in metrics or calculate it
        
        if "wfo_test" in metrics:
            st.metric("Test IC", f"{metrics['wfo_test'].get('ic', 0):.3f}")
            st.metric("Test Sharpe", f"{metrics['wfo_test'].get('sharpe', 0):.2f}")
            
        # Rolling IC Chart
        # We can compute it from preds
        ic_series = preds.groupby("date").apply(lambda x: x["label"].corr(x["proba"]))
        
        fig_ic = px.line(ic_series, title="Rolling Information Coefficient (IC)")
        fig_ic.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_ic, use_container_width=True)
        
        st.metric("Positive IC Months", f"{(ic_series > 0).mean():.1%}")

    with tab3:
        st.header("Alpha Drivers")
        if not fi.empty:
            fig_fi = px.bar(fi.head(20), x="importance", y="Unnamed: 0", orientation='h', title="Top 20 Features")
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("No feature importance data available.")

if __name__ == "__main__":
    main()
