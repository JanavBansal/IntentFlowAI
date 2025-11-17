"""Inference pipeline for surfacing swing opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from intentflow_ai.features import FeatureEngineer
from intentflow_ai.meta_labeling import MetaLabelConfig, MetaLabeler
from intentflow_ai.modeling import ExplanationConfig, RegimeClassifier, explain_signals
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScoringPipeline:
    """Apply trained models to incoming price panels and emit ranked signals."""

    feature_engineer: FeatureEngineer
    models: Dict[str, Any]
    feature_columns: Optional[List[str]] = None
    regime_classifier: Optional[RegimeClassifier] = None
    default_model_key: str = "overall"
    meta_model: Any | None = None
    meta_feature_columns: Optional[List[str]] = None
    meta_config: Optional[MetaLabelConfig] = None
    explanation_config: Optional[ExplanationConfig] = None
    background_data: Optional[pd.DataFrame] = None

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            raise ValueError("Scoring dataset is empty.")
        features = self.feature_engineer.build(dataset).fillna(0.0)
        if self.feature_columns:
            features = features.reindex(columns=self.feature_columns, fill_value=0.0)

        assignments = self._regime_assignments(dataset)
        proba = pd.Series(index=dataset.index, dtype=float)
        for regime, idx in assignments.items():
            model = self.models.get(regime) or self.models.get(self.default_model_key)
            if model is None:
                continue
            subset = features.loc[idx]
            if subset.empty:
                continue
            preds = model.predict_proba(subset)[:, 1]
            proba.loc[idx] = preds

        signals = dataset[["date", "ticker"]].copy()
        if "sector" in dataset.columns:
            signals["sector"] = dataset["sector"]
        signals["proba"] = proba
        signals.index = dataset.index  # Preserve original index for SHAP alignment
        
        if self.meta_model is not None and self.meta_feature_columns and self.meta_config:
            meta_labeler = MetaLabeler(self.meta_config)
            try:
                meta_proba = meta_labeler.predict(self.meta_model, dataset, proba, self.meta_feature_columns)
                signals[self.meta_config.output_col] = meta_proba
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning("Meta-model scoring failed", exc_info=exc)
        
        signals["rank"] = signals["proba"].rank(ascending=False, method="first")
        
        # Add SHAP explanations before sorting (to preserve index alignment)
        if self.explanation_config and self.explanation_config.enabled and self.background_data is not None:
            try:
                # Use the default model for explanations
                model = self.models.get(self.default_model_key)
                if model is not None and not self.background_data.empty:
                    # Generate explanations with aligned indices
                    signals_with_explanations = explain_signals(
                        model=model,
                        features=features,
                        signals=signals,
                        background_data=self.background_data,
                        config=self.explanation_config,
                    )
                    signals = signals_with_explanations
                    logger.info("Added SHAP explanations to signals", extra={"count": len(signals)})
                else:
                    logger.warning("Model or background data unavailable for SHAP")
            except Exception as exc:
                logger.warning("SHAP explanation failed, returning signals without explanations", exc_info=exc)
        
        # Sort and reset index after explanations
        signals = signals.sort_values("rank").reset_index(drop=True)
        
        logger.info("Generated signals", extra={"count": len(signals)})
        return signals

    def _regime_assignments(self, dataset: pd.DataFrame) -> Dict[str, pd.Index]:
        if (
            not self.regime_classifier
            or "date" not in dataset.columns
            or "close" not in dataset.columns
        ):
            return {self.default_model_key: dataset.index}

        market_series = dataset.groupby("date")["close"].mean().sort_index()
        regime_map = self.regime_classifier.infer(market_series).to_dict()

        dates = pd.to_datetime(dataset["date"])
        regimes = pd.Series(
            [regime_map.get(date, self.default_model_key) for date in dates],
            index=dataset.index,
        )
        assignments: Dict[str, pd.Index] = {}
        for regime_name, idx in regimes.groupby(regimes).groups.items():
            assignments[regime_name] = idx
        return assignments

    def explain(self) -> Dict[str, Any]:
        return {
            "model_keys": list(self.models.keys()),
            "feature_blocks": list(self.feature_engineer.feature_blocks.keys()),
        }
