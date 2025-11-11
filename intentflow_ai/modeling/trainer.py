"""LightGBM training utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import lightgbm as lgb
import pandas as pd

from intentflow_ai.config.settings import LightGBMConfig
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LightGBMTrainer:
    """Thin wrapper over LightGBM's sklearn API with telemetry hooks."""

    cfg: LightGBMConfig

    def train(self, features: pd.DataFrame, target: pd.Series) -> lgb.LGBMClassifier:
        logger.info("Fitting LightGBM", extra={"rows": len(features), "cols": features.shape[1]})
        model = lgb.LGBMClassifier(**asdict(self.cfg))
        model.fit(features, target)
        return model

    def feature_importance(self, model: lgb.LGBMClassifier) -> Dict[str, float]:
        importances = dict(zip(model.feature_name_, model.feature_importances_.tolist()))
        logger.debug("Computed feature importances", extra={"count": len(importances)})
        return importances

    def predict_with_meta_label(self, model: lgb.LGBMClassifier, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        proba = pd.Series(model.predict_proba(features)[:, 1], index=features.index)
        meta = (proba > 0.6).astype(int)
        return proba, meta
