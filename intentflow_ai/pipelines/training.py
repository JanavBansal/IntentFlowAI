"""End-to-end training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.features import FeatureEngineer
from intentflow_ai.modeling import LightGBMTrainer, ModelEvaluator
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingPipeline:
    """Wire together feature engineering, LightGBM, and evaluation."""

    cfg: Settings = settings
    feature_engineer: FeatureEngineer = FeatureEngineer()

    def run(self, dataset: pd.DataFrame, target: pd.Series) -> dict:
        features = self.feature_engineer.build(dataset)
        trainer = LightGBMTrainer(self.cfg.lgbm)
        model = trainer.train(features, target)
        proba, meta = trainer.predict_with_meta_label(model, features)

        evaluator = ModelEvaluator(horizon_days=self.cfg.signal_horizon_days)
        metrics = evaluator.evaluate(target, proba)
        logger.info("Training complete", extra={"metrics": metrics})
        return {
            "model": model,
            "probabilities": proba,
            "meta_labels": meta,
            "metrics": metrics,
        }
