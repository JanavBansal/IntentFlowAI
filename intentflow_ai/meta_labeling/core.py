"""Lightweight meta-labeling scaffold built atop primary model signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.config.settings import LightGBMConfig
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetaLabelConfig:
    """Configuration for meta-label training and inference."""

    enabled: bool = False
    horizon_days: int = 10
    success_threshold: float = 0.02
    min_signal_proba: float = 0.0
    random_state: int = 42
    proba_col: str = "proba"
    output_col: str = "meta_proba"


@dataclass
class MetaLabeler:
    """Train a secondary classifier to decide whether to take trades."""

    cfg: MetaLabelConfig = field(default_factory=MetaLabelConfig)
    _trainer: LightGBMTrainer = field(init=False)

    def __post_init__(self) -> None:
        meta_lgbm = LightGBMConfig(
            random_state=self.cfg.random_state,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.8,
            subsample=0.8,
            subsample_freq=1,
            max_depth=-1,
        )
        self._trainer = LightGBMTrainer(meta_lgbm)

    def build_training_frame(self, frame: pd.DataFrame, base_proba: pd.Series) -> pd.DataFrame:
        """Combine base signals with context features and realized outcomes."""

        fwd_col = f"fwd_ret_{self.cfg.horizon_days}d"
        if fwd_col not in frame.columns:
            raise ValueError(f"Training frame missing forward return column '{fwd_col}'.")
        df = frame.copy()
        df[self.cfg.proba_col] = base_proba
        features = self._build_features(df)
        label = (df[fwd_col] >= self.cfg.success_threshold).astype(int)
        train_df = features.copy()
        train_df["meta_label"] = label
        if self.cfg.min_signal_proba > 0:
            keep = train_df[self.cfg.proba_col] >= self.cfg.min_signal_proba
            train_df = train_df.loc[keep]
        train_df = train_df.dropna(subset=["meta_label"])
        return train_df

    def train(self, frame: pd.DataFrame, base_proba: pd.Series) -> Dict[str, Any]:
        """Fit the meta-model and return predictions on the input frame."""

        train_df = self.build_training_frame(frame, base_proba)
        if train_df.empty:
            raise ValueError("No rows available for meta-label training.")

        feature_cols = [col for col in train_df.columns if col != "meta_label"]
        X = train_df[feature_cols]
        y = train_df["meta_label"]
        model = self._trainer.train(X, y)
        proba, _ = self._trainer.predict_with_meta_label(model, X)
        logger.info(
            "Trained meta-model",
            extra={
                "rows": len(train_df),
                "features": len(feature_cols),
                "success_rate": float(y.mean()),
            },
        )
        return {"model": model, "proba": proba, "feature_columns": feature_cols, "labels": y}

    def predict(
        self,
        model: Any,
        frame: pd.DataFrame,
        base_proba: pd.Series,
        feature_columns: List[str],
    ) -> pd.Series:
        """Generate meta probabilities for new samples."""

        df = frame.copy()
        df[self.cfg.proba_col] = base_proba
        features = self._build_features(df)
        features = features.reindex(columns=feature_columns, fill_value=0.0)
        preds, _ = self._trainer.predict_with_meta_label(model, features)
        return preds

    def _build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Construct lightweight context features for the meta-model."""

        features = pd.DataFrame(index=frame.index)
        features[self.cfg.proba_col] = frame[self.cfg.proba_col]

        if {"ticker", "close", "date"}.issubset(frame.columns):
            grouped = frame.groupby("ticker", group_keys=False)
            pct = grouped["close"].transform(lambda s: s.pct_change())
            features["ret_5d"] = grouped["close"].transform(lambda s: s.pct_change(5))
            features["vol_20"] = pct.rolling(20).std()
            features["drawdown_20"] = grouped["close"].transform(
                lambda s: (s / s.rolling(20).max()) - 1.0
            )

            dates = pd.to_datetime(frame["date"])
            market = frame.groupby(dates)["close"].mean().sort_index()
            market_ret = market.pct_change()
            market_vol = market_ret.rolling(20).std()
            trend_fast = market.rolling(50).mean()
            trend_slow = market.rolling(200).mean()
            trend_signal = (trend_fast / (trend_slow + 1e-9)) - 1.0

            features["market_vol_20"] = dates.map(market_vol)
            features["market_trend_50_200"] = dates.map(trend_signal)

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features
