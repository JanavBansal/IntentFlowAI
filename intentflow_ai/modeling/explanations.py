"""SHAP-based feature explanations for trading signals.

This module provides interpretability tools to explain model predictions
at the position level, enabling audit and understanding of signal rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExplanationConfig:
    """Configuration for SHAP explanations."""

    enabled: bool = True
    max_features: int = 10  # Top N features to include in explanations
    background_samples: int = 100  # Samples for SHAP background
    use_tree_explainer: bool = True  # Use TreeExplainer (faster) vs KernelExplainer


class SHAPExplainer:
    """Generate SHAP values for LightGBM model predictions."""

    def __init__(self, config: ExplanationConfig | None = None):
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is required for explanations. Install with: pip install shap"
            )
        self.config = config or ExplanationConfig()
        self.explainer: Any = None
        self.background_data: pd.DataFrame | None = None

    def fit(self, model: Any, background_data: pd.DataFrame) -> None:
        """Fit the SHAP explainer with background data."""
        if not self.config.enabled:
            return

        logger.info(
            "Fitting SHAP explainer",
            extra={"samples": len(background_data), "features": background_data.shape[1]},
        )

        # Sample background data if too large
        if len(background_data) > self.config.background_samples:
            background_data = background_data.sample(
                n=self.config.background_samples, random_state=42
            )

        self.background_data = background_data

        if self.config.use_tree_explainer:
            try:
                self.explainer = shap.TreeExplainer(model, background_data)
            except Exception as exc:
                logger.warning(
                    "TreeExplainer failed, falling back to KernelExplainer", exc_info=exc
                )
                self.explainer = shap.KernelExplainer(
                    model.predict_proba, background_data.iloc[:50]
                )
        else:
            self.explainer = shap.KernelExplainer(
                model.predict_proba, background_data.iloc[:50]
            )

    def explain(
        self, features: pd.DataFrame, model: Any | None = None
    ) -> pd.DataFrame:
        """Generate SHAP values for given features.

        Returns:
            DataFrame with same index as features, columns are feature names,
            values are SHAP contributions to positive class probability.
        """
        if not self.config.enabled or self.explainer is None:
            return pd.DataFrame(index=features.index)

        if features.empty:
            return pd.DataFrame(index=features.index)

        logger.debug("Computing SHAP values", extra={"rows": len(features)})

        try:
            shap_values = self.explainer.shap_values(features)
            # For binary classification, shap_values is a list [class_0, class_1]
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Use positive class
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 1]  # Positive class

            shap_df = pd.DataFrame(
                shap_values, index=features.index, columns=features.columns
            )
            return shap_df
        except Exception as exc:
            logger.error("SHAP computation failed", exc_info=exc)
            return pd.DataFrame(index=features.index)

    def get_top_features(
        self, shap_values: pd.DataFrame, top_k: int | None = None
    ) -> pd.DataFrame:
        """Extract top contributing features for each prediction.

        Args:
            shap_values: DataFrame of SHAP values (from explain())
            top_k: Number of top features to return (default: config.max_features)

        Returns:
            DataFrame with columns: feature_name, shap_value, abs_shap_value
            Indexed by original row index, with one row per top feature per prediction
        """
        if shap_values.empty:
            return pd.DataFrame()

        top_k = top_k or self.config.max_features
        results = []

        for idx in shap_values.index:
            row = shap_values.loc[idx]
            abs_values = row.abs().sort_values(ascending=False)
            top_features = abs_values.head(top_k)

            for feature, abs_shap in top_features.items():
                shap_val = row[feature]
                results.append(
                    {
                        "index": idx,
                        "feature_name": feature,
                        "shap_value": shap_val,
                        "abs_shap_value": abs_shap,
                    }
                )

        return pd.DataFrame(results)

    def summarize_signal(
        self,
        ticker: str,
        date: pd.Timestamp,
        proba: float,
        shap_values: pd.Series,
        features: pd.Series,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        """Generate human-readable explanation for a single signal.

        Returns:
            Dictionary with signal explanation including:
            - ticker, date, probability
            - top contributing features with values and SHAP contributions
            - rationale summary
        """
        top_k = top_k or self.config.max_features
        abs_shap = shap_values.abs().sort_values(ascending=False)
        top_features = abs_shap.head(top_k)

        explanation = {
            "ticker": ticker,
            "date": date.isoformat() if isinstance(date, pd.Timestamp) else str(date),
            "probability": float(proba),
            "top_features": [],
            "rationale": "",
        }

        for feature in top_features.index:
            shap_val = shap_values[feature]
            feat_val = features.get(feature, np.nan)
            explanation["top_features"].append(
                {
                    "feature": feature,
                    "value": float(feat_val) if not pd.isna(feat_val) else None,
                    "shap_contribution": float(shap_val),
                    "abs_contribution": float(abs(shap_val)),
                }
            )

        # Generate simple rationale
        positive_features = [
            f["feature"]
            for f in explanation["top_features"]
            if f["shap_contribution"] > 0
        ]
        negative_features = [
            f["feature"]
            for f in explanation["top_features"]
            if f["shap_contribution"] < 0
        ]

        rationale_parts = []
        if positive_features:
            rationale_parts.append(
                f"Boosted by: {', '.join(positive_features[:3])}"
            )
        if negative_features:
            rationale_parts.append(
                f"Suppressed by: {', '.join(negative_features[:3])}"
            )

        explanation["rationale"] = ". ".join(rationale_parts) if rationale_parts else "No strong feature drivers identified."

        return explanation


def explain_signals(
    model: Any,
    features: pd.DataFrame,
    signals: pd.DataFrame,
    background_data: pd.DataFrame,
    config: ExplanationConfig | None = None,
) -> pd.DataFrame:
    """Generate SHAP explanations for a set of signals.

    Args:
        model: Trained LightGBM model
        features: Feature matrix for signals (must align with dataset used to create signals)
        signals: DataFrame with columns: date, ticker, proba, rank (index may be reset)
        background_data: Background dataset for SHAP
        config: Explanation configuration

    Returns:
        DataFrame with explanations, one row per signal, including:
        - All original signal columns
        - top_features (list of dicts)
        - rationale (string)
        - shap_values (dict mapping feature -> SHAP value)
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping explanations")
        return signals.copy()

    config = config or ExplanationConfig()
    explainer = SHAPExplainer(config)
    explainer.fit(model, background_data)

    # Generate SHAP values for all features
    shap_df = explainer.explain(features)
    if shap_df.empty:
        logger.warning("SHAP values empty, returning signals without explanations")
        return signals.copy()

    # Align SHAP values with signals using index
    # Signals should have same index as features (before sorting)
    results = signals.copy()
    explanations = []

    # Match by index - signals and features should have aligned indices
    for idx in signals.index:
        if idx not in shap_df.index:
            explanations.append({})
            continue

        shap_row = shap_df.loc[idx]
        signal_row = signals.loc[idx]
        feature_row = features.loc[idx] if idx in features.index else pd.Series()

        explanation = explainer.summarize_signal(
            ticker=str(signal_row.get("ticker", "")),
            date=pd.to_datetime(signal_row.get("date", pd.Timestamp.now())),
            proba=float(signal_row.get("proba", 0.0)),
            shap_values=shap_row,
            features=feature_row,
            top_k=config.max_features,
        )

        explanations.append(explanation)

    # Add explanation columns
    results["top_features"] = [e.get("top_features", []) for e in explanations]
    results["rationale"] = [e.get("rationale", "") for e in explanations]
    results["shap_values"] = [
        {f["feature"]: f["shap_contribution"] for f in e.get("top_features", [])}
        for e in explanations
    ]

    return results

