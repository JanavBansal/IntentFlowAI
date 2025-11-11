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
