"""End-to-end training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
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
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger
from intentflow_ai.utils.splits import purged_time_series_splits, time_splits

logger = get_logger(__name__)


@dataclass
class TrainingPipeline:
    """Wire together feature engineering, LightGBM, and evaluation."""

    cfg: Settings = field(default_factory=lambda: settings)
    feature_engineer: FeatureEngineer = field(default_factory=FeatureEngineer)
    regime_filter: bool = True
    use_live_sources: bool = False
    leak_test: bool = False

    def run(
        self,
        *,
        live: bool | None = None,
        tickers_subset: Optional[Sequence[str]] = None,
        leak_test: bool | None = None,
    ) -> dict:
        live_mode = self.use_live_sources if live is None else live
        if live_mode:
            DataIngestionWorkflow().run()
        leak_mode = self.leak_test if leak_test is None else leak_test

        price_panel = load_price_parquet(allow_fallback=False)
        if tickers_subset is not None:
            price_panel = price_panel[price_panel["ticker"].isin(tickers_subset)]
            if price_panel.empty:
                raise ValueError("No price data left after ticker filtering.")
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

        train_mask, valid_mask, test_mask = self._build_splits(train_df)
        train_ticker_count = train_df.loc[train_mask, "ticker"].nunique()
        if train_ticker_count < self.cfg.min_train_tickers:
            raise AssertionError(
                f"Training tickers {train_ticker_count} below minimum {self.cfg.min_train_tickers}. "
                "Ensure full universe parquet is available."
            )

        features = train_df[feature_cols]
        target = train_df["label"].copy()
        if leak_mode:
            rng = np.random.default_rng(self.cfg.lgbm_seed)
            shuffled = pd.Series(rng.permutation(target.values), index=target.index)
            train_df["label_true"] = target
            train_df["label"] = shuffled
            target = shuffled
            logger.warning("Leak test enabled; labels were shuffled before training.")
        train_valid_mask = train_mask | valid_mask
        if not train_valid_mask.any():
            raise ValueError("Need at least one row in train/valid set.")

        trainer = LightGBMTrainer(self.cfg.lgbm)
        overall_model = trainer.train(features.loc[train_valid_mask], target.loc[train_valid_mask])
        overall_proba, _ = trainer.predict_with_meta_label(overall_model, features)
        train_df["proba"] = overall_proba

        metrics: Dict[str, Dict[str, float]] = {
            "overall": self._compute_metrics(train_df, overall_proba),
            "train": self._compute_metrics(train_df, overall_proba, train_mask),
            "valid": self._compute_metrics(train_df, overall_proba, valid_mask),
            "test": self._compute_metrics(train_df, overall_proba, test_mask),
        }
        feature_importances = trainer.feature_importance(overall_model)

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
                        split_metrics[split_name] = self._compute_metrics(train_df.loc[idx], scores.loc[idx])
                if split_metrics:
                    by_regime[regime_name] = split_metrics
            if by_regime:
                metrics["by_regime"] = by_regime

        cv_metrics = self._purged_cv_metrics(
            trainer,
            train_df,
            features,
            target,
        )
        if cv_metrics:
            metrics["purged_cv"] = cv_metrics

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
                "train": train_mask.tolist(),
                "valid": valid_mask.tolist(),
                "test": test_mask.tolist(),
            },
            "tickers_used": sorted(train_df["ticker"].unique()),
            "leak_test": leak_mode,
            "feature_importances": feature_importances,
            "cv_metrics": cv_metrics,
        }

    def _compute_metrics(
        self,
        frame: pd.DataFrame,
        scores: pd.Series,
        mask: Optional[pd.Series] = None,
    ) -> Dict[str, object]:
        if mask is not None:
            idx = frame.index[mask]
            if idx.size == 0:
                return self._empty_metrics()
            subset = frame.loc[idx]
            subset_scores = scores.loc[idx]
        else:
            subset = frame
            subset_scores = scores
            if subset.empty:
                return self._empty_metrics()

        evaluator = ModelEvaluator(horizon_days=self.cfg.signal_horizon_days)
        try:
            base = evaluator.evaluate(
                subset["label"],
                subset_scores,
                excess_returns=subset.get("excess_fwd"),
                dates=subset.get("date"),
            )
        except ValueError:
            base = {}

        result = self._empty_metrics()
        result.update(base)
        result["precision_at_10"] = precision_at_k(subset["label"], subset_scores, 10)
        result["precision_at_20"] = precision_at_k(subset["label"], subset_scores, 20)
        result["hit_rate_0.6"] = hit_rate(subset["label"], subset_scores, thresh=0.6)
        return result

    @staticmethod
    def _empty_metrics() -> Dict[str, object]:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "precision_at_10": float("nan"),
            "precision_at_20": float("nan"),
            "hit_rate_0.6": float("nan"),
            "precision_by_day_at_10": float("nan"),
            "precision_by_day_at_20": float("nan"),
            "ic": float("nan"),
            "rank_ic": float("nan"),
            "hit_curve": [],
            "horizon_days": float("nan"),
        }

    def _purged_cv_metrics(
        self,
        trainer: LightGBMTrainer,
        frame: pd.DataFrame,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> list[dict]:
        try:
            splits = purged_time_series_splits(
                frame,
                date_col="date",
                n_splits=self.cfg.cv_splits,
                embargo_days=self.cfg.signal_horizon_days,
                horizon_days=self.cfg.signal_horizon_days,
            )
        except ValueError as exc:
            logger.warning("Skipping purged CV", extra={"reason": str(exc)})
            return []

        cv_metrics: list[dict] = []
        for idx, (train_mask, test_mask) in enumerate(splits, start=1):
            if not train_mask.any() or not test_mask.any():
                continue
            model = trainer.train(features.loc[train_mask], target.loc[train_mask])
            proba, _ = trainer.predict_with_meta_label(model, features)
            metrics = self._compute_metrics(frame, proba, mask=test_mask)
            metrics["fold"] = idx
            cv_metrics.append(metrics)
        return cv_metrics

    def _build_splits(self, train_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        try:
            return time_splits(
                train_df,
                date_col="date",
                valid_start=self.cfg.valid_start,
                test_start=self.cfg.test_start,
                embargo_days=self.cfg.signal_horizon_days,
                horizon_days=self.cfg.signal_horizon_days,
            )
        except ValueError as exc:
            logger.warning(
                "Primary date splits invalid (%s); deriving dynamic splits from available data.",
                exc,
            )
            dates = pd.to_datetime(train_df["date"]).sort_values().reset_index(drop=True)
            if dates.empty:
                raise
            valid_idx = max(int(len(dates) * 0.6), 1)
            test_idx = max(int(len(dates) * 0.8), valid_idx + 1)
            valid_idx = min(valid_idx, len(dates) - 2)
            test_idx = min(test_idx, len(dates) - 1)
            dyn_valid = dates.iloc[valid_idx]
            dyn_test = dates.iloc[test_idx]
            return time_splits(
                train_df,
                date_col="date",
                valid_start=str(dyn_valid.date()),
                test_start=str(dyn_test.date()),
                embargo_days=self.cfg.signal_horizon_days,
                horizon_days=self.cfg.signal_horizon_days,
            )
