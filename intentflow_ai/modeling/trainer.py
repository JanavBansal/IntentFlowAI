"""LightGBM training utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except ModuleNotFoundError as exc:  # pragma: no cover - fallback path
    raise ModuleNotFoundError(
        "LightGBM is required. Please install it via `pip install lightgbm` before running the pipeline."
    ) from exc

from intentflow_ai.config.settings import LightGBMConfig
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LightGBMTrainer:
    """Thin wrapper over LightGBM's sklearn API with telemetry hooks."""

    cfg: LightGBMConfig

    def train(self, features: pd.DataFrame, target: pd.Series) -> lgb.LGBMClassifier:
        logger.info("Fitting LightGBM", extra={"rows": len(features), "cols": features.shape[1]})
        params = asdict(self.cfg)
        model = lgb.LGBMClassifier(**params)
        model.fit(features, target)
        return model

    def feature_importance(self, model) -> Dict[str, float]:
        names = getattr(model, "feature_name_", None)
        values = getattr(model, "feature_importances_", None)
        if names is None or values is None:
            return {}
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [float(values)] * len(names)
        else:
            values = np.asarray(values).tolist()
        importances = dict(zip(names, values))
        logger.debug("Computed feature importances", extra={"count": len(importances)})
        return importances

    def predict_with_meta_label(self, model, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        proba = pd.Series(model.predict_proba(features)[:, 1], index=features.index)
        meta = (proba > 0.6).astype(int)
        return proba, meta
