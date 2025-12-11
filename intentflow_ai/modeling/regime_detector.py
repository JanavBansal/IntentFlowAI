"""
Regime Detection Module

Implements Hidden Markov Model (HMM) based market regime detection.
Identifies market states: Risk-On, Risk-Off, Uncertain.

Key signals:
- India VIX level and changes
- Market breadth (% stocks above 50 DMA)
- FII cumulative flow direction

Usage:
    detector = RegimeDetector(n_regimes=3)
    detector.fit(historical_features)
    current_regime = detector.predict_regime(current_features)
    if detector.should_trade():
        # Execute strategy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime states."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    UNCERTAIN = "uncertain"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    n_regimes: int = 3
    vix_threshold_high: float = 25.0  # India VIX threshold for high volatility
    vix_threshold_low: float = 15.0   # India VIX threshold for low volatility
    breadth_threshold_high: float = 0.60  # % stocks above 50 DMA for bullish
    breadth_threshold_low: float = 0.40   # % stocks above 50 DMA for bearish
    fii_lookback_days: int = 5        # Days for FII flow momentum
    min_training_samples: int = 252   # Minimum samples for HMM training (1 year)


@dataclass
class RegimeDetector:
    """
    Hidden Markov Model based market regime detector.
    
    Identifies distinct market regimes to enable regime-conditional trading:
    - Risk-On: Favorable conditions, proceed with predictions
    - Risk-Off: Adverse conditions, reduce exposure
    - Uncertain: Mixed signals, trade with caution
    """
    
    config: RegimeConfig = field(default_factory=RegimeConfig)
    _model: Optional[object] = field(default=None, init=False)
    _regime_labels: Dict[int, MarketRegime] = field(default_factory=dict, init=False)
    _current_regime: MarketRegime = field(default=MarketRegime.UNCERTAIN, init=False)
    _is_fitted: bool = field(default=False, init=False)
    
    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        """
        Fit HMM on historical regime features.
        
        Args:
            features: DataFrame with columns:
                - vix: India VIX level
                - vix_change: Daily change in VIX
                - breadth: % of stocks above 50 DMA
                - fii_flow: Cumulative FII flow (5-day)
                
        Returns:
            self (fitted detector)
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed. Using rule-based regime detection.")
            self._is_fitted = True
            return self
        
        if len(features) < self.config.min_training_samples:
            logger.warning(
                f"Insufficient samples for HMM training: {len(features)} < {self.config.min_training_samples}. "
                "Using rule-based fallback."
            )
            self._is_fitted = True
            return self
        
        # Prepare features for HMM
        feature_cols = self._get_feature_columns(features)
        X = features[feature_cols].dropna().values
        
        if len(X) < self.config.min_training_samples:
            logger.warning("Too many NaN values in features. Using rule-based fallback.")
            self._is_fitted = True
            return self
        
        # Fit Gaussian HMM
        self._model = hmm.GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        try:
            self._model.fit(X)
            self._label_regimes(features, feature_cols)
            self._is_fitted = True
            logger.info(
                f"Regime detector fitted successfully. "
                f"Regimes: {list(self._regime_labels.values())}"
            )
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}. Using rule-based fallback.")
            self._model = None
            self._is_fitted = True
        
        return self
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get available feature columns for HMM."""
        possible_cols = ["vix", "vix_change", "breadth", "fii_flow"]
        return [c for c in possible_cols if c in df.columns]
    
    def _label_regimes(self, features: pd.DataFrame, feature_cols: List[str]) -> None:
        """
        Label HMM states based on observed characteristics.
        
        Strategy: 
        - Regime with lowest VIX mean → Risk-On
        - Regime with highest VIX mean → Risk-Off
        - Remaining → Uncertain
        """
        X = features[feature_cols].dropna().values
        states = self._model.predict(X)
        
        # Calculate mean VIX for each state
        if "vix" in feature_cols:
            vix_idx = feature_cols.index("vix")
            state_vix_means = {}
            for state in range(self.config.n_regimes):
                state_mask = states == state
                if state_mask.any():
                    state_vix_means[state] = X[state_mask, vix_idx].mean()
                else:
                    state_vix_means[state] = float('inf')
            
            # Sort states by VIX mean
            sorted_states = sorted(state_vix_means.items(), key=lambda x: x[1])
            
            self._regime_labels = {
                sorted_states[0][0]: MarketRegime.RISK_ON,      # Lowest VIX
                sorted_states[-1][0]: MarketRegime.RISK_OFF,    # Highest VIX
            }
            
            # Label remaining as uncertain
            for state in range(self.config.n_regimes):
                if state not in self._regime_labels:
                    self._regime_labels[state] = MarketRegime.UNCERTAIN
        else:
            # Default labeling if VIX not available
            for state in range(self.config.n_regimes):
                self._regime_labels[state] = MarketRegime.UNCERTAIN
    
    def predict_regime(self, features: pd.DataFrame) -> MarketRegime:
        """
        Predict current market regime.
        
        Args:
            features: Current market features (single row or recent history)
            
        Returns:
            MarketRegime enum value
        """
        if not self._is_fitted:
            logger.warning("Regime detector not fitted. Returning UNCERTAIN.")
            return MarketRegime.UNCERTAIN
        
        # Try HMM prediction first
        if self._model is not None:
            try:
                feature_cols = self._get_feature_columns(features)
                X = features[feature_cols].dropna().values
                if len(X) > 0:
                    state = self._model.predict(X)[-1]  # Latest state
                    self._current_regime = self._regime_labels.get(state, MarketRegime.UNCERTAIN)
                    return self._current_regime
            except Exception as e:
                logger.warning(f"HMM prediction failed: {e}. Using rule-based fallback.")
        
        # Rule-based fallback
        return self._rule_based_regime(features)
    
    def _rule_based_regime(self, features: pd.DataFrame) -> MarketRegime:
        """
        Simple rule-based regime detection (fallback).
        
        Rules:
        - VIX > 25 AND breadth < 40% → Risk-Off
        - VIX < 15 AND breadth > 60% → Risk-On
        - Otherwise → Uncertain
        """
        if features.empty:
            return MarketRegime.UNCERTAIN
        
        latest = features.iloc[-1] if len(features) > 1 else features.iloc[0]
        
        vix = latest.get("vix", 20.0)
        breadth = latest.get("breadth", 0.50)
        
        if vix > self.config.vix_threshold_high and breadth < self.config.breadth_threshold_low:
            self._current_regime = MarketRegime.RISK_OFF
        elif vix < self.config.vix_threshold_low and breadth > self.config.breadth_threshold_high:
            self._current_regime = MarketRegime.RISK_ON
        else:
            self._current_regime = MarketRegime.UNCERTAIN
        
        return self._current_regime
    
    def should_trade(self) -> bool:
        """
        Determine if trading is recommended in current regime.
        
        Returns:
            True if regime is Risk-On, False otherwise
        """
        return self._current_regime == MarketRegime.RISK_ON
    
    def get_regime_confidence(self) -> float:
        """
        Get confidence score for current regime prediction.
        
        Returns:
            Float between 0.0 and 1.0
        """
        if self._model is None:
            return 0.5  # Neutral confidence for rule-based
        
        # Could use posterior probabilities from HMM
        # For now, return fixed confidence based on regime
        confidence_map = {
            MarketRegime.RISK_ON: 0.8,
            MarketRegime.RISK_OFF: 0.8,
            MarketRegime.UNCERTAIN: 0.5
        }
        return confidence_map.get(self._current_regime, 0.5)
    
    def get_regime_summary(self) -> Dict:
        """Get summary of current regime state."""
        return {
            "regime": self._current_regime.value,
            "should_trade": self.should_trade(),
            "confidence": self.get_regime_confidence(),
            "is_hmm_fitted": self._model is not None
        }


def compute_regime_features(
    prices_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    fii_df: Optional[pd.DataFrame] = None,
    lookback: int = 50
) -> pd.DataFrame:
    """
    Compute features required for regime detection.
    
    Args:
        prices_df: Price data for universe (columns: date, ticker, close)
        vix_df: India VIX data (columns: date, close)
        fii_df: FII flow data (columns: date, fii_net)
        lookback: Lookback period for moving average
        
    Returns:
        DataFrame with regime features indexed by date
    """
    features = pd.DataFrame()
    
    # Get unique dates
    if "date" in prices_df.columns:
        dates = prices_df["date"].unique()
        features["date"] = pd.to_datetime(dates)
        features = features.set_index("date").sort_index()
    
    # Compute market breadth (% stocks above 50 DMA)
    if "close" in prices_df.columns and "ticker" in prices_df.columns:
        breadth_series = _compute_market_breadth(prices_df, lookback)
        features["breadth"] = breadth_series
    
    # Add VIX features
    if vix_df is not None and not vix_df.empty:
        vix_df = vix_df.set_index("date") if "date" in vix_df.columns else vix_df
        features["vix"] = vix_df["close"]
        features["vix_change"] = features["vix"].pct_change()
        features["vix_5d_change"] = features["vix"].pct_change(5)
    
    # Add FII flow features
    if fii_df is not None and not fii_df.empty:
        fii_df = fii_df.set_index("date") if "date" in fii_df.columns else fii_df
        features["fii_flow"] = fii_df["fii_net"].rolling(5).sum()
        features["fii_direction"] = np.sign(fii_df["fii_net"])
    
    return features.dropna()


def _compute_market_breadth(prices_df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Compute market breadth: % of stocks above their N-day moving average.
    
    Args:
        prices_df: Price data (columns: date, ticker, close)
        lookback: Moving average period
        
    Returns:
        Series with breadth values indexed by date
    """
    # Compute MA for each ticker
    prices_df = prices_df.copy()
    prices_df["ma"] = prices_df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(lookback, min_periods=lookback).mean()
    )
    
    # Check if price > MA
    prices_df["above_ma"] = (prices_df["close"] > prices_df["ma"]).astype(int)
    
    # Compute daily breadth
    breadth = prices_df.groupby("date").agg(
        above_ma_count=("above_ma", "sum"),
        total_count=("above_ma", "count")
    )
    breadth["breadth"] = breadth["above_ma_count"] / breadth["total_count"]
    
    return breadth["breadth"]
