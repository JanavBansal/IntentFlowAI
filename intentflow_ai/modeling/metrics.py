"""Additional evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(y_true: pd.Series, y_score: pd.Series, k: int) -> float:
    """Return precision among the top-k scored observations."""

    if k <= 0:
        raise ValueError("k must be positive.")
    if y_true.empty:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(-y_score.values)
    top_idx = order[:k]
    return float(y_true.iloc[top_idx].mean())


def hit_rate(y_true: pd.Series, y_score: pd.Series, thresh: float) -> float:
    """Fraction of rows that are true positives above a score threshold."""

    if y_true.empty:
        return 0.0
    hits = (y_score >= thresh) & (y_true == 1)
    return float(hits.mean())


def stability_report(
    df: pd.DataFrame,
    date_col: str = "date",
    label_col: str = "label",
    proba_col: str = "proba",
    freq: str = "M",
) -> dict:
    """Compute stability metrics like IC mean, IC std, and win rate over time."""
    if df.empty:
        return {}
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Compute IC per period
    ic_series = df.groupby(pd.Grouper(key=date_col, freq=freq)).apply(
        lambda x: x[label_col].corr(x[proba_col]) if len(x) > 1 else np.nan
    )
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_sharpe = ic_mean / ic_std if ic_std > 0 else 0.0
    positive_periods = (ic_series > 0).mean()
    
    return {
        "ic_mean": float(ic_mean) if not pd.isna(ic_mean) else 0.0,
        "ic_std": float(ic_std) if not pd.isna(ic_std) else 0.0,
        "ic_sharpe": float(ic_sharpe),
        "positive_period_rate": float(positive_periods),
    }
