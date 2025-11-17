"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn import metrics

from intentflow_ai.modeling.metrics import hit_rate
from intentflow_ai.modeling.trading_metrics import (
    compute_decile_ic,
    compute_return_ic,
    compute_sharpe_by_decile,
)


@dataclass
class ModelEvaluator:
    """Compute classification metrics and custom finance diagnostics."""

    horizon_days: int

    def evaluate(
        self,
        y_true: pd.Series,
        y_score: pd.Series,
        *,
        excess_returns: pd.Series | None = None,
        dates: pd.Series | None = None,
    ) -> Dict[str, float | List[dict]]:
        auc = metrics.roc_auc_score(y_true, y_score)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        pr_auc = metrics.auc(recall, precision)
        base_hit_rate = float(np.mean((y_score > 0.6) & (y_true == 1)))

        payload: Dict[str, float | List[dict]] = {
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
            "hit_rate": base_hit_rate,
            "horizon_days": float(self.horizon_days),
        }

        if excess_returns is not None:
            aligned = pd.DataFrame(
                {
                    "ret": excess_returns,
                    "score": y_score,
                }
            ).dropna()
            if not aligned.empty:
                # Return IC (Pearson correlation of signal magnitude with returns)
                payload["return_ic"] = compute_return_ic(aligned["score"], aligned["ret"])
                # Rank IC (Spearman correlation)
                payload["rank_ic"] = float(aligned["ret"].corr(aligned["score"], method="spearman"))
                # Legacy: keep "ic" as alias for return_ic
                payload["ic"] = payload["return_ic"]

                # Decile analysis
                decile_ic, decile_stats = compute_decile_ic(aligned["score"], aligned["ret"])
                payload["decile_ic"] = decile_ic
                payload["decile_stats"] = decile_stats.to_dict("records") if not decile_stats.empty else []

                # Sharpe by decile
                sharpe_by_decile = compute_sharpe_by_decile(aligned["score"], aligned["ret"])
                payload["sharpe_by_decile"] = sharpe_by_decile
            else:
                payload["return_ic"] = float("nan")
                payload["ic"] = float("nan")
                payload["rank_ic"] = float("nan")
                payload["decile_ic"] = float("nan")
                payload["decile_stats"] = []
                payload["sharpe_by_decile"] = []
        else:
            payload["return_ic"] = float("nan")
            payload["ic"] = float("nan")
            payload["rank_ic"] = float("nan")
            payload["decile_ic"] = float("nan")
            payload["decile_stats"] = []
            payload["sharpe_by_decile"] = []

        if dates is not None:
            day_metrics = self._precision_by_day(y_true, y_score, pd.to_datetime(dates), 10)
            payload["precision_by_day_at_10"] = day_metrics
            payload["precision_by_day_at_20"] = self._precision_by_day(y_true, y_score, pd.to_datetime(dates), 20)
        else:
            payload["precision_by_day_at_10"] = float("nan")
            payload["precision_by_day_at_20"] = float("nan")

        thresholds = np.round(np.arange(0.5, 0.91, 0.05), 2)
        curve = []
        for thresh in thresholds:
            curve.append(
                {
                    "threshold": float(thresh),
                    "hit_rate": hit_rate(y_true, y_score, thresh),
                    "coverage": float((y_score >= thresh).mean()),
                }
            )
        payload["hit_curve"] = curve
        return payload

    def _precision_by_day(self, y_true: pd.Series, y_score: pd.Series, dates: pd.Series, k: int) -> float:
        frame = pd.DataFrame({"date": dates, "label": y_true, "score": y_score}).dropna()
        if frame.empty:
            return float("nan")
        values = []
        for _, group in frame.groupby("date"):
            ranked = group.sort_values("score", ascending=False).head(k)
            if not ranked.empty:
                values.append(float(ranked["label"].mean()))
        return float(np.mean(values)) if values else float("nan")
