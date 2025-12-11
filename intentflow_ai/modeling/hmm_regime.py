"""Hidden Markov Model based Market Regime Detection.

Implements a 4-state HMM for detecting market regimes:
1. Bull Market (trending up, low volatility)
2. Bear Market (trending down, high volatility)  
3. High Volatility Sideways (range-bound, high volatility)
4. Low Volatility Sideways (range-bound, low volatility)

References:
- State Street Global Advisors: Regime Detection using ML (2024)
- QuestDB: Market Regime Detection with HMM
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime states."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL_SIDEWAYS = "high_vol_sideways"
    LOW_VOL_SIDEWAYS = "low_vol_sideways"


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    regime: MarketRegime
    probability: float
    transition_probs: Dict[str, float]
    feature_values: Dict[str, float]
    detection_date: datetime


class HMMRegimeDetector:
    """4-state Hidden Markov Model for market regime detection.
    
    Features used for regime detection:
    - Trailing returns (20d, 60d)
    - Realized volatility (20d)
    - Volatility of volatility
    - Trend strength (MA crossovers)
    - RSI regime indicator
    
    The model adapts trading strategy based on detected regime:
    - Bull: Momentum strategies work best
    - Bear: Mean reversion, defensive
    - High Vol: Reduce position sizes
    - Low Vol: Breakout strategies
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        lookback_days: int = 252,  # 1 year
        min_history_days: int = 60  # Minimum history for regime detection
    ):
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days
        self._hmm = None
        self._is_fitted = False
        self._scaler = None
        
    def _check_hmmlearn(self) -> bool:
        """Check if hmmlearn is available."""
        try:
            import hmmlearn
            return True
        except ImportError:
            logger.warning("hmmlearn not available. Install with: pip install hmmlearn")
            return False
    
    def _compute_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute features for regime detection.
        
        Args:
            market_data: DataFrame with 'date' and 'close' columns
            
        Returns:
            DataFrame with regime detection features
        """
        df = market_data.copy()
        df = df.sort_values('date')
        close = df['close']
        
        # Returns at different horizons
        df['ret_5d'] = close.pct_change(5)
        df['ret_20d'] = close.pct_change(20)
        df['ret_60d'] = close.pct_change(60)
        
        # Realized volatility
        daily_ret = close.pct_change()
        df['vol_20d'] = daily_ret.rolling(20).std() * np.sqrt(252)  # Annualized
        df['vol_60d'] = daily_ret.rolling(60).std() * np.sqrt(252)
        
        # Volatility of volatility (regime uncertainty)
        df['vol_of_vol'] = df['vol_20d'].rolling(20).std()
        
        # Trend strength (MA ratio)
        df['ma_20'] = close.rolling(20).mean()
        df['ma_60'] = close.rolling(60).mean()
        df['trend_strength'] = (df['ma_20'] / df['ma_60']) - 1
        
        # RSI for overbought/oversold regime
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Distance from 52-week high/low
        df['high_252d'] = close.rolling(252, min_periods=60).max()
        df['low_252d'] = close.rolling(252, min_periods=60).min()
        df['dist_from_high'] = close / df['high_252d'] - 1
        df['dist_from_low'] = close / df['low_252d'] - 1
        
        return df
    
    def fit(self, market_data: pd.DataFrame) -> 'HMMRegimeDetector':
        """Fit HMM to historical market data.
        
        Args:
            market_data: DataFrame with 'date' and 'close' columns
        """
        if not self._check_hmmlearn():
            self._is_fitted = False
            return self
        
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
        
        # Compute features
        df = self._compute_features(market_data)
        
        # Select feature columns for HMM
        feature_cols = ['ret_20d', 'vol_20d', 'trend_strength', 'dist_from_high']
        X = df[feature_cols].dropna()
        
        if len(X) < self.min_history_days:
            logger.warning(f"Insufficient data for HMM fitting: {len(X)} days")
            self._is_fitted = False
            return self
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Fit HMM
        self._hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        self._hmm.fit(X_scaled)
        
        # Store feature columns for inference
        self._feature_cols = feature_cols
        self._is_fitted = True
        
        logger.info(f"HMM fitted with {self.n_regimes} regimes on {len(X)} observations")
        
        return self
    
    def detect_regime(self, market_data: pd.DataFrame) -> RegimeState:
        """Detect current market regime.
        
        Args:
            market_data: DataFrame with 'date' and 'close' columns
            
        Returns:
            RegimeState with current regime and probabilities
        """
        if not self._is_fitted:
            # Fallback to rule-based detection
            return self._rule_based_detection(market_data)
        
        # Compute features
        df = self._compute_features(market_data)
        X = df[self._feature_cols].dropna()
        
        if len(X) < 5:
            return self._rule_based_detection(market_data)
        
        # Scale features
        X_scaled = self._scaler.transform(X)
        
        # Get state probabilities
        state_probs = self._hmm.predict_proba(X_scaled)
        current_state = state_probs[-1]  # Latest observation
        regime_idx = np.argmax(current_state)
        
        # Map state index to regime
        regime = self._map_state_to_regime(regime_idx, df.iloc[-1])
        
        # Get transition probabilities
        trans_probs = self._hmm.transmat_[regime_idx]
        
        return RegimeState(
            regime=regime,
            probability=float(current_state[regime_idx]),
            transition_probs={
                MarketRegime.BULL.value: float(trans_probs[0]),
                MarketRegime.BEAR.value: float(trans_probs[1]) if len(trans_probs) > 1 else 0,
                MarketRegime.HIGH_VOL_SIDEWAYS.value: float(trans_probs[2]) if len(trans_probs) > 2 else 0,
                MarketRegime.LOW_VOL_SIDEWAYS.value: float(trans_probs[3]) if len(trans_probs) > 3 else 0,
            },
            feature_values={
                'ret_20d': float(df['ret_20d'].iloc[-1]) if pd.notna(df['ret_20d'].iloc[-1]) else 0,
                'vol_20d': float(df['vol_20d'].iloc[-1]) if pd.notna(df['vol_20d'].iloc[-1]) else 0,
                'trend_strength': float(df['trend_strength'].iloc[-1]) if pd.notna(df['trend_strength'].iloc[-1]) else 0,
            },
            detection_date=df['date'].iloc[-1] if 'date' in df.columns else datetime.now()
        )
    
    def _map_state_to_regime(self, state_idx: int, latest_features: pd.Series) -> MarketRegime:
        """Map HMM state index to regime using feature characteristics.
        
        Uses the learned state's feature means to determine regime type.
        """
        if not self._is_fitted or self._hmm is None:
            return MarketRegime.LOW_VOL_SIDEWAYS
        
        # Get state means
        state_means = self._hmm.means_[state_idx]
        
        # Interpret based on scaled feature values
        # [ret_20d, vol_20d, trend_strength, dist_from_high]
        ret_mean = state_means[0]
        vol_mean = state_means[1]
        trend_mean = state_means[2]
        
        # Classification logic
        if ret_mean > 0.5 and trend_mean > 0:
            return MarketRegime.BULL
        elif ret_mean < -0.5 or trend_mean < -0.3:
            return MarketRegime.BEAR
        elif vol_mean > 0.5:
            return MarketRegime.HIGH_VOL_SIDEWAYS
        else:
            return MarketRegime.LOW_VOL_SIDEWAYS
    
    def _rule_based_detection(self, market_data: pd.DataFrame) -> RegimeState:
        """Fallback rule-based regime detection when HMM not available."""
        df = self._compute_features(market_data)
        
        if df.empty:
            return RegimeState(
                regime=MarketRegime.LOW_VOL_SIDEWAYS,
                probability=0.5,
                transition_probs={r.value: 0.25 for r in MarketRegime},
                feature_values={},
                detection_date=datetime.now()
            )
        
        latest = df.iloc[-1]
        
        ret_20d = latest.get('ret_20d', 0) if pd.notna(latest.get('ret_20d')) else 0
        vol_20d = latest.get('vol_20d', 0.15) if pd.notna(latest.get('vol_20d')) else 0.15
        trend = latest.get('trend_strength', 0) if pd.notna(latest.get('trend_strength')) else 0
        
        # Rule-based classification
        # High volatility threshold: annualized vol > 25%
        is_high_vol = vol_20d > 0.25
        
        # Trend detection
        is_uptrend = ret_20d > 0.05 and trend > 0.02
        is_downtrend = ret_20d < -0.05 or trend < -0.02
        
        if is_uptrend and not is_high_vol:
            regime = MarketRegime.BULL
            prob = 0.7
        elif is_downtrend or (is_high_vol and ret_20d < 0):
            regime = MarketRegime.BEAR
            prob = 0.7
        elif is_high_vol:
            regime = MarketRegime.HIGH_VOL_SIDEWAYS
            prob = 0.6
        else:
            regime = MarketRegime.LOW_VOL_SIDEWAYS
            prob = 0.6
        
        return RegimeState(
            regime=regime,
            probability=prob,
            transition_probs={r.value: 0.25 for r in MarketRegime},
            feature_values={
                'ret_20d': float(ret_20d),
                'vol_20d': float(vol_20d),
                'trend_strength': float(trend),
            },
            detection_date=df['date'].iloc[-1] if 'date' in df.columns else datetime.now()
        )
    
    def get_regime_strategy_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get recommended strategy weights for a regime.
        
        Returns weights for different alpha sources based on regime.
        """
        weights = {
            MarketRegime.BULL: {
                'momentum': 0.4,
                'mean_reversion': 0.1,
                'sector_relative': 0.3,
                'quality': 0.2,
            },
            MarketRegime.BEAR: {
                'momentum': 0.1,
                'mean_reversion': 0.3,
                'sector_relative': 0.2,
                'quality': 0.4,
            },
            MarketRegime.HIGH_VOL_SIDEWAYS: {
                'momentum': 0.15,
                'mean_reversion': 0.35,
                'sector_relative': 0.25,
                'quality': 0.25,
            },
            MarketRegime.LOW_VOL_SIDEWAYS: {
                'momentum': 0.25,
                'mean_reversion': 0.25,
                'sector_relative': 0.25,
                'quality': 0.25,
            },
        }
        return weights.get(regime, weights[MarketRegime.LOW_VOL_SIDEWAYS])


def get_regime_detector() -> HMMRegimeDetector:
    """Factory function to get regime detector."""
    return HMMRegimeDetector()
