"""Inference pipeline for surfacing swing opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from intentflow_ai.features import FeatureEngineer
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScoringPipeline:
    """Apply a trained model to fresh features and emit ranked signals."""

    feature_engineer: FeatureEngineer
    model: Any

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        features = self.feature_engineer.build(dataset)
        proba = self.model.predict_proba(features)[:, 1]
        signals = pd.DataFrame(
            {
                "proba": proba,
                "rank": pd.Series(proba).rank(ascending=False),
            },
            index=dataset.index,
        )
        logger.info("Generated signals", extra={"count": len(signals)})
        return signals.sort_values("rank")

    def explain(self) -> Dict[str, Any]:
        """Return metadata about the scoring configuration."""

        return {
            "model_class": type(self.model).__name__,
            "feature_blocks": list(self.feature_engineer.feature_blocks.keys()),
        }
