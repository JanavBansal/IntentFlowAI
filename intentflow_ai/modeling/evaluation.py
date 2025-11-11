"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import metrics


@dataclass
class ModelEvaluator:
    """Compute classification metrics and custom finance diagnostics."""

    horizon_days: int

    def evaluate(self, y_true: pd.Series, y_score: pd.Series) -> Dict[str, float]:
        auc = metrics.roc_auc_score(y_true, y_score)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        pr_auc = metrics.auc(recall, precision)
        hit_rate = float(np.mean((y_score > 0.6) & (y_true == 1)))
        return {
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
            "hit_rate": hit_rate,
            "horizon_days": float(self.horizon_days),
        }
