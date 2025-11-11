"""End-to-end training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.data.ingestion import DataIngestionWorkflow
from intentflow_ai.features import FeatureEngineer, make_excess_label
from intentflow_ai.modeling import (
    LightGBMTrainer,
    ModelEvaluator,
    RegimeClassifier,
    hit_rate,
    precision_at_k,
)
from intentflow_ai.utils import time_splits
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
        label_cols = ["label", "excess_fwd", f"fwd_ret_{self.cfg.signal_horizon_days}d", "sector_fwd"]
        train_df = labeled.dropna(subset=feature_cols + label_cols).reset_index(drop=True)
        if train_df.empty:
            raise ValueError("Training dataset is empty after dropping NA rows.")

        train_mask, valid_mask, test_mask = time_splits(
            train_df,
            date_col="date",
            valid_start=self.cfg.valid_start,
            test_start=self.cfg.test_start,
        )

        features = train_df[feature_cols]
        target = train_df["label"]
        train_valid_mask = train_mask | valid_mask
        if not train_valid_mask.any():
            raise ValueError("Need at least one row in train/valid set.")

        trainer = LightGBMTrainer(self.cfg.lgbm)
        overall_model = trainer.train(features.loc[train_valid_mask], target.loc[train_valid_mask])
        overall_proba, _ = trainer.predict_with_meta_label(overall_model, features)

        metrics: Dict[str, Dict[str, float]] = {
            "overall": self._compute_metrics(target, overall_proba),
            "train": self._compute_metrics(target.loc[train_mask], overall_proba.loc[train_mask]),
            "valid": self._compute_metrics(target.loc[valid_mask], overall_proba.loc[valid_mask]),
            "test": self._compute_metrics(target.loc[test_mask], overall_proba.loc[test_mask]),
        }

        regime_classifier = None
        models = {"overall": overall_model}
        regime_scores: Dict[str, pd.Series] = {}
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
                regime_scores[regime] = regime_proba

            by_regime: Dict[str, Dict[str, Dict[str, float]]] = {}
            for regime_name, scores in regime_scores.items():
                regime_mask = train_df["regime"] == regime_name
                split_metrics: Dict[str, Dict[str, float]] = {}
                for split_name, mask in {
                    "train": train_mask,
                    "valid": valid_mask,
                    "test": test_mask,
                }.items():
                    subset_mask = regime_mask & mask
                    if subset_mask.any():
                        idx = train_df.index[subset_mask]
                        split_metrics[split_name] = self._compute_metrics(
                            target.loc[idx],
                            scores.loc[idx],
                        )
                if split_metrics:
                    by_regime[regime_name] = split_metrics
            if by_regime:
                metrics["by_regime"] = by_regime

        logger.info("Training complete", extra={"metrics": metrics})
        return {
            "model": overall_model,
            "models": models,
            "probabilities": overall_proba,
            "metrics": metrics,
            "training_frame": train_df,
            "feature_columns": feature_cols,
            "regime_classifier": regime_classifier,
            "splits": {
                "train": train_mask,
                "valid": valid_mask,
                "test": test_mask,
            },
        }

    def _compute_metrics(self, y_true: pd.Series, y_score: pd.Series) -> Dict[str, float]:
        if y_true.empty:
            return {
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "precision_at_10": float("nan"),
                "precision_at_20": float("nan"),
                "hit_rate_0.6": float("nan"),
            }

        evaluator = ModelEvaluator(horizon_days=self.cfg.signal_horizon_days)
        try:
            base = evaluator.evaluate(y_true, y_score)
            roc_auc = base.get("roc_auc", float("nan"))
            pr_auc = base.get("pr_auc", float("nan"))
        except ValueError:
            roc_auc = float("nan")
            pr_auc = float("nan")

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_at_10": precision_at_k(y_true, y_score, 10),
            "precision_at_20": precision_at_k(y_true, y_score, 20),
            "hit_rate_0.6": hit_rate(y_true, y_score, thresh=0.6),
        }
