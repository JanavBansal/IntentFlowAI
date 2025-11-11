"""End-to-end training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.data.ingestion import DataIngestionWorkflow
from intentflow_ai.features import FeatureEngineer, make_excess_label
from intentflow_ai.modeling import LightGBMTrainer, ModelEvaluator, RegimeClassifier
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingPipeline:
    """Wire together feature engineering, LightGBM, and evaluation."""

    cfg: Settings = field(default_factory=lambda: settings)
    feature_engineer: FeatureEngineer = field(default_factory=FeatureEngineer)
    regime_filter: bool = True
    use_live_sources: bool = False

    def run(self, live: bool | None = None) -> dict:
        live_mode = self.use_live_sources if live is None else live
        if live_mode:
            DataIngestionWorkflow().run()

        price_panel = load_price_parquet()
        feature_frame = self.feature_engineer.build(price_panel)
        dataset = price_panel.join(feature_frame)
        labeled = make_excess_label(
            dataset,
            horizon_days=self.cfg.signal_horizon_days,
            thresh=self.cfg.target_excess_return,
        )

        feature_cols = feature_frame.columns.tolist()
        required_cols = feature_cols + ["label", "excess_fwd"]
        train_df = labeled.dropna(subset=required_cols).reset_index(drop=True)
        if train_df.empty:
            raise ValueError("Training dataset is empty after dropping NA rows.")

        features = train_df[feature_cols]
        target = train_df["label"]

        trainer = LightGBMTrainer(self.cfg.lgbm)
        overall_model = trainer.train(features, target)
        overall_proba, _ = trainer.predict_with_meta_label(overall_model, features)

        evaluator = ModelEvaluator(horizon_days=self.cfg.signal_horizon_days)
        metrics = {"overall": evaluator.evaluate(target, overall_proba), "regimes": {}}

        regime_classifier = None
        models = {"overall": overall_model}
        if self.regime_filter:
            regime_classifier = RegimeClassifier()
            market_series = price_panel.groupby("date")["close"].mean().sort_index()
            regime_map = regime_classifier.infer(market_series)
            train_df["date"] = pd.to_datetime(train_df["date"])
            train_df = train_df.merge(regime_map.rename("regime"), left_on="date", right_index=True, how="left")

            for regime, subset in train_df.dropna(subset=["regime"]).groupby("regime"):
                if subset.empty:
                    continue
                subset_features = subset[feature_cols]
                subset_target = subset["label"]
                regime_model = trainer.train(subset_features, subset_target)
                models[regime] = regime_model
                regime_proba, _ = trainer.predict_with_meta_label(regime_model, subset_features)
                metrics["regimes"][regime] = evaluator.evaluate(subset_target, regime_proba)

        logger.info("Training complete", extra={"metrics": metrics})
        return {
            "model": overall_model,
            "models": models,
            "probabilities": overall_proba,
            "metrics": metrics,
            "training_frame": train_df,
            "feature_columns": feature_cols,
            "regime_classifier": regime_classifier,
        }
