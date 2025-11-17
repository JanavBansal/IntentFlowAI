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
        """Construct lightweight context features for the meta-model.
        
        Includes base model prediction, delivery/flow features, risk metrics, and regime indicators.
        """

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
        
        # Add delivery/flow features if available (from feature engineering)
        # Look for delivery features in the frame (they may be prefixed with "delivery__")
        delivery_cols = [col for col in frame.columns if "deliv" in col.lower() or "delivery" in col.lower()]
        for col in delivery_cols:
            # Extract the base feature name (remove prefix if present)
            base_name = col.split("__")[-1] if "__" in col else col
            # Use short names for meta-model
            if base_name in ["deliv_ratio", "deliv_ratio_mean_10", "deliv_value_spike", "deliv_vs_price_mom_10"]:
                features[f"meta_{base_name}"] = frame[col]
        
        # Add sector-relative features if available
        sector_cols = [col for col in frame.columns if "sector" in col.lower() and ("mom" in col.lower() or "vol" in col.lower())]
        for col in sector_cols:
            base_name = col.split("__")[-1] if "__" in col else col
            if "z" in base_name:  # Only include z-scores
                features[f"meta_{base_name}"] = frame[col]
        
        # Add regime/volatility features if available
        regime_cols = [col for col in frame.columns if "regime" in col.lower() or "index_vol" in col.lower()]
        for col in regime_cols:
            base_name = col.split("__")[-1] if "__" in col else col
            features[f"meta_{base_name}"] = frame[col]
        
        # Add ownership/structural features if available
        ownership_cols = [col for col in frame.columns if any(x in col.lower() for x in ["fii", "dii", "promoter", "pledged", "free_float"])]
        for col in ownership_cols:
            base_name = col.split("__")[-1] if "__" in col else col
            features[f"meta_{base_name}"] = frame[col]

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features


@dataclass
class EnhancedMetaLabelConfig(MetaLabelConfig):
    """Enhanced configuration with drawdown and win rate focus."""
    
    # Drawdown control
    max_stock_drawdown: float = -0.15  # Block trades if stock down >15%
    max_portfolio_drawdown: float = -0.10  # Stop trading if portfolio down >10%
    
    # Win rate targeting
    target_win_rate: float = 0.55  # Target 55% win rate
    min_risk_reward: float = 1.5  # Minimum risk/reward ratio
    
    # Position sizing
    use_kelly_sizing: bool = False
    max_position_size: float = 0.05  # Max 5% per position
    
    # Regime filtering
    block_high_vol_regime: bool = True
    vol_threshold_pct: float = 2.0  # Block if market vol > 2%
    
    # Historical performance filtering
    use_similar_pattern_filter: bool = True
    min_pattern_success_rate: float = 0.50  # Only trade if similar patterns succeeded


@dataclass
class EnhancedMetaLabeler(MetaLabeler):
    """Enhanced meta-labeler with advanced risk controls."""
    
    cfg: EnhancedMetaLabelConfig = field(default_factory=EnhancedMetaLabelConfig)
    
    def _build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Construct enhanced features including risk metrics."""
        # Get base features
        features = super()._build_features(frame)
        
        if {"ticker", "close", "date"}.issubset(frame.columns):
            grouped = frame.groupby("ticker", group_keys=False)
            
            # Enhanced risk features
            
            # 1. Drawdown metrics
            rolling_max = grouped["close"].transform(lambda s: s.rolling(60).max())
            features["drawdown_60d"] = (frame["close"] / rolling_max) - 1.0
            features["drawdown_severity"] = features["drawdown_60d"].clip(upper=0).abs()
            
            # 2. Win rate proxies (historical success in similar conditions)
            features["ret_10d_lag"] = grouped["close"].transform(lambda s: s.pct_change(10).shift(10))
            features["success_lag"] = (features["ret_10d_lag"] > 0.015).astype(float)
            
            # Rolling success rate (historical win rate)
            features["rolling_win_rate_20"] = (
                features["success_lag"].rolling(20).mean()
            )
            
            # 3. Volatility regime
            ret_series = grouped["close"].transform(lambda s: s.pct_change())
            features["vol_regime_z"] = (
                (ret_series.rolling(20).std() - ret_series.rolling(60).std().mean()) /
                ret_series.rolling(60).std().std()
            )
            
            # 4. Momentum quality (steady vs choppy)
            features["momentum_quality"] = (
                features["ret_5d"] / (ret_series.rolling(5).std() + 1e-9)
            )
            
            # 5. Risk/reward proxy
            # Expected upside vs downside
            upside = grouped["close"].transform(
                lambda s: (s.rolling(20).max() - s) / s
            )
            downside = grouped["close"].transform(
                lambda s: (s - s.rolling(20).min()) / s
            )
            features["risk_reward_ratio"] = upside / (downside + 0.01)
            
            # 6. Market regime features
            dates = pd.to_datetime(frame["date"])
            market = frame.groupby(dates)["close"].mean().sort_index()
            market_ret = market.pct_change()
            
            # Market stress indicator
            market_stress = market_ret.rolling(10).std() / market_ret.rolling(60).std()
            features["market_stress"] = dates.map(market_stress)
            
            # Market breadth (what % of stocks are rising)
            stock_rets = grouped["close"].transform(lambda s: s.pct_change())
            breadth = frame.groupby("date")["close"].transform(lambda g: (g.pct_change() > 0).mean())
            features["market_breadth"] = breadth
            
            # 7. Correlation with market (diversification benefit)
            def rolling_corr_with_market(ticker_series):
                """Compute rolling correlation with market."""
                ticker_rets = ticker_series.pct_change()
                # Align dates
                aligned = pd.DataFrame({
                    "ticker_ret": ticker_rets,
                    "market_ret": market_ret.reindex(ticker_rets.index)
                }).dropna()
                
                if len(aligned) < 20:
                    return pd.Series(0.0, index=ticker_series.index)
                
                rolling_corr = aligned["ticker_ret"].rolling(20).corr(aligned["market_ret"])
                return rolling_corr.reindex(ticker_series.index, fill_value=0.0)
            
            features["market_correlation"] = grouped["close"].transform(rolling_corr_with_market)
        
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features
    
    def apply_risk_filters(
        self,
        frame: pd.DataFrame,
        base_proba: pd.Series,
        meta_proba: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Apply hard risk filters to block risky trades.
        
        Returns mask of allowed trades.
        """
        if frame.empty:
            return pd.Series(True, index=frame.index)
        
        allowed = pd.Series(True, index=frame.index)
        
        # Build features for filter rules
        features = self._build_features(frame.copy())
        features[self.cfg.proba_col] = base_proba
        if meta_proba is not None:
            features["meta_proba"] = meta_proba
        
        # Filter 1: Block large drawdowns
        if "drawdown_60d" in features.columns:
            allowed &= features["drawdown_60d"] > self.cfg.max_stock_drawdown
        
        # Filter 2: Block high volatility regime
        if self.cfg.block_high_vol_regime and "market_stress" in features.columns:
            allowed &= features["market_stress"] < 2.0  # 2x normal volatility
        
        # Filter 3: Require positive historical win rate
        if "rolling_win_rate_20" in features.columns:
            allowed &= features["rolling_win_rate_20"] > self.cfg.min_pattern_success_rate
        
        # Filter 4: Require positive risk/reward
        if "risk_reward_ratio" in features.columns:
            allowed &= features["risk_reward_ratio"] > self.cfg.min_risk_reward
        
        # Filter 5: Avoid extreme negative momentum
        if "ret_5d" in features.columns:
            allowed &= features["ret_5d"] > -0.05  # Don't catch falling knives
        
        # Filter 6: Require sufficient market breadth
        if "market_breadth" in features.columns:
            allowed &= features["market_breadth"] > 0.3  # At least 30% stocks rising
        
        blocked_count = (~allowed).sum()
        logger.info(
            "Applied risk filters",
            extra={
                "total_signals": len(allowed),
                "blocked": blocked_count,
                "allowed": allowed.sum(),
                "block_rate": f"{blocked_count / len(allowed) * 100:.1f}%",
            }
        )
        
        return allowed
    
    def compute_position_sizes(
        self,
        frame: pd.DataFrame,
        base_proba: pd.Series,
        meta_proba: pd.Series,
        *,
        total_capital: float = 1.0,
    ) -> pd.Series:
        """Compute position sizes using Kelly criterion or volatility-based sizing.
        
        Returns position sizes as fraction of capital.
        """
        if not self.cfg.use_kelly_sizing:
            # Equal weight with max cap
            position_sizes = pd.Series(self.cfg.max_position_size, index=frame.index)
            return position_sizes
        
        # Kelly criterion: f* = (p*b - q) / b
        # where p = win probability, q = 1-p, b = win/loss ratio
        
        features = self._build_features(frame.copy())
        
        # Estimate win probability from meta_proba
        win_prob = meta_proba
        
        # Estimate risk/reward ratio
        risk_reward = features.get("risk_reward_ratio", pd.Series(1.5, index=frame.index))
        risk_reward = risk_reward.clip(lower=1.0, upper=5.0)  # Reasonable bounds
        
        # Kelly fraction
        kelly_fraction = (win_prob * risk_reward - (1 - win_prob)) / risk_reward
        kelly_fraction = kelly_fraction.clip(lower=0.0, upper=self.cfg.max_position_size)
        
        # Apply volatility adjustment (lower size in high vol)
        if "vol_20" in features.columns:
            vol_adj = 1.0 / (1.0 + features["vol_20"] * 10)  # Reduce size with volatility
            kelly_fraction = kelly_fraction * vol_adj
        
        # Final cap
        position_sizes = kelly_fraction.clip(upper=self.cfg.max_position_size)
        
        logger.info(
            "Computed Kelly position sizes",
            extra={
                "mean_size": f"{position_sizes.mean():.2%}",
                "max_size": f"{position_sizes.max():.2%}",
            }
        )
        
        return position_sizes
