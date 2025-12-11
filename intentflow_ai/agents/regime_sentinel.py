"""
Regime Sentinel Agent - Agent 3 of Council of Experts.

Detects market regime using Hidden Markov Model (HMM).
Regimes: Bull, Bear, High-Vol Sideways, Low-Vol Sideways

This agent refactors the existing HMMRegimeDetector from V3
into the new agent interface.

The regime influences how other agents' signals are weighted
in the final debate.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from intentflow_ai.agents.base_agent import BaseAgent, AgentOutput
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeSentinelAgent(BaseAgent):
    """
    Agent 3: Regime Sentinel
    
    Uses a 4-state Hidden Markov Model to detect market regime:
    - Bull: Trending up, low volatility
    - Bear: Trending down, high volatility
    - High-Vol Sideways: Range-bound, high volatility
    - Low-Vol Sideways: Range-bound, low volatility
    
    The detected regime affects:
    1. Signal direction (+0.5 Bull, -0.5 Bear, 0 Sideways)
    2. Confidence based on regime probability
    3. Other agents' weight in debate (momentum stronger in Bull, etc.)
    """
    
    # Regime constants
    BULL = 0
    BEAR = 1
    HIGH_VOL = 2
    LOW_VOL = 3
    
    REGIME_NAMES = {
        0: "Bull",
        1: "Bear",
        2: "High-Vol Sideways",
        3: "Low-Vol Sideways"
    }
    
    REGIME_SIGNALS = {
        0: 0.5,   # Bull: positive bias
        1: -0.5,  # Bear: negative bias
        2: 0.0,   # High-vol: neutral (stay out)
        3: 0.1,   # Low-vol: slight positive (mean reversion works)
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Regime Sentinel", config=config or {})
        
        self.n_regimes = self.config.get('n_regimes', 4)
        self.lookback_days = self.config.get('lookback_days', 252)
        
        # HMM parameters
        self._hmm_model = None
        self._hmm_available = self._check_hmmlearn()
        
        # State mapping (learned from data)
        self._state_to_regime = {}
    
    def _check_hmmlearn(self) -> bool:
        """Check if hmmlearn is installed."""
        try:
            from hmmlearn import hmm
            return True
        except ImportError:
            logger.warning("hmmlearn not installed. Will use rule-based regime detection.")
            return False
    
    def get_required_features(self) -> List[str]:
        """Features for regime detection."""
        return [
            # Returns
            'momentum__price_ret_20d',  # 20-day return
            
            # Volatility
            'volatility__price_vol_20',  # 20-day realized vol
            'volatility__vol_ratio_short_long',  # Vol regime change
            
            # Market features (if available)
            'india_vix',  # VIX equivalent
            'market_return_20d',  # NIFTY return
        ]
    
    def _compute_regime_features(self, X: pd.DataFrame) -> np.ndarray:
        """Extract features for HMM from raw data."""
        features = []
        
        # Return (required)
        if 'momentum__price_ret_20d' in X.columns:
            features.append(X['momentum__price_ret_20d'].values)
        elif 'close' in X.columns:
            # Compute 20d return from close prices
            ret = X['close'].pct_change(20).values
            features.append(ret)
        else:
            features.append(np.zeros(len(X)))
        
        # Volatility (required)
        if 'volatility__price_vol_20' in X.columns:
            features.append(X['volatility__price_vol_20'].values)
        elif 'close' in X.columns:
            # Compute rolling volatility
            vol = X['close'].pct_change().rolling(20).std().values
            features.append(vol)
        else:
            features.append(np.zeros(len(X)))
        
        # Stack and handle NaNs
        features_array = np.column_stack(features)
        features_array = np.nan_to_num(features_array, nan=0.0)
        
        return features_array
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> None:
        """
        Train HMM on historical market data.
        
        Note: HMM is unsupervised - y_train is ignored.
        """
        if not self._hmm_available:
            logger.info("Using rule-based regime detection (no HMM)")
            self.is_trained = True
            return
        
        from hmmlearn import hmm
        
        features = self._compute_regime_features(X_train)
        
        # Remove rows with all zeros (insufficient data)
        valid_mask = ~(features == 0).all(axis=1)
        features = features[valid_mask]
        
        if len(features) < 100:
            logger.warning("Insufficient data for HMM training. Using rule-based detection.")
            self.is_trained = True
            return
        
        logger.info(f"Training {self.name} HMM with {len(features)} samples")
        
        # Fit HMM
        self._hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        try:
            self._hmm_model.fit(features)
            
            # Map HMM states to regime labels using means
            self._map_states_to_regimes(features)
            
            self.is_trained = True
            logger.info("HMM training complete")
            
        except Exception as e:
            logger.warning(f"HMM training failed: {e}. Using rule-based detection.")
            self._hmm_model = None
            self.is_trained = True
    
    def _map_states_to_regimes(self, features: np.ndarray) -> None:
        """Map HMM hidden states to interpretable regime labels."""
        if self._hmm_model is None:
            return
        
        # Get state means
        means = self._hmm_model.means_
        
        # features[:, 0] = return, features[:, 1] = volatility
        ret_means = means[:, 0]
        vol_means = means[:, 1] if means.shape[1] > 1 else np.zeros(self.n_regimes)
        
        # Classify each state
        for state in range(self.n_regimes):
            ret_val = ret_means[state]
            vol_val = vol_means[state]
            
            if ret_val > 0.01:  # Positive return
                if vol_val < np.median(vol_means):
                    self._state_to_regime[state] = self.BULL
                else:
                    self._state_to_regime[state] = self.HIGH_VOL
            else:  # Negative/flat return
                if vol_val > np.median(vol_means):
                    self._state_to_regime[state] = self.BEAR
                else:
                    self._state_to_regime[state] = self.LOW_VOL
    
    def predict(self, X: pd.DataFrame) -> AgentOutput:
        """Detect current market regime."""
        if not self.is_trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        # Try HMM prediction first
        if self._hmm_model is not None:
            return self._hmm_predict(X)
        else:
            return self._rule_based_predict(X)
    
    def _hmm_predict(self, X: pd.DataFrame) -> AgentOutput:
        """Predict using trained HMM."""
        features = self._compute_regime_features(X)
        
        # Get most likely state sequence
        try:
            state_sequence = self._hmm_model.predict(features)
            state_probs = self._hmm_model.predict_proba(features)
            
            current_state = state_sequence[-1]
            current_probs = state_probs[-1]
            
            regime = self._state_to_regime.get(current_state, self.LOW_VOL)
            regime_name = self.REGIME_NAMES[regime]
            
            signal = self.REGIME_SIGNALS[regime]
            confidence = float(current_probs[current_state])
            
            # Build transition info
            transition_info = {
                self.REGIME_NAMES[self._state_to_regime.get(i, i)]: f"{p:.0%}"
                for i, p in enumerate(current_probs)
            }
            
            reasoning = f"Market regime: {regime_name} (confidence: {confidence:.0%}). Transitions: {transition_info}"
            
            return AgentOutput(
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                feature_attribution={"regime": regime, "regime_name": regime_name},
                agent_name=self.name
            )
            
        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}. Falling back to rule-based.")
            return self._rule_based_predict(X)
    
    def _rule_based_predict(self, X: pd.DataFrame) -> AgentOutput:
        """Fallback rule-based regime detection."""
        # Get return
        ret = 0.0
        if 'momentum__price_ret_20d' in X.columns:
            ret = float(X['momentum__price_ret_20d'].iloc[-1])
        
        # Get volatility
        vol = 0.02  # Default
        if 'volatility__price_vol_20' in X.columns:
            vol = float(X['volatility__price_vol_20'].iloc[-1])
        
        # Classify regime
        high_vol = vol > 0.025
        positive_ret = ret > 0.01
        
        if positive_ret and not high_vol:
            regime = self.BULL
        elif not positive_ret and high_vol:
            regime = self.BEAR
        elif high_vol:
            regime = self.HIGH_VOL
        else:
            regime = self.LOW_VOL
        
        regime_name = self.REGIME_NAMES[regime]
        signal = self.REGIME_SIGNALS[regime]
        confidence = 0.6  # Lower confidence for rule-based
        
        reasoning = f"Market regime (rule-based): {regime_name} | Return: {ret:.1%}, Vol: {vol:.1%}"
        
        return AgentOutput(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            feature_attribution={"regime": regime, "regime_name": regime_name},
            agent_name=self.name
        )
    
    def get_regime_weights(self, regime: int = None) -> Dict[str, float]:
        """
        Get recommended agent weights for a given regime.
        
        Used by the Council to adjust debate weights based on market conditions.
        """
        weights = {
            self.BULL: {
                "Technical Analyst": 0.35,
                "Flow Detective": 0.30,
                "Regime Sentinel": 0.10,
                "Risk Contrarian": 0.10,
                "Earnings Oracle": 0.15,
            },
            self.BEAR: {
                "Technical Analyst": 0.20,
                "Flow Detective": 0.25,
                "Regime Sentinel": 0.10,
                "Risk Contrarian": 0.35,  # Higher in bear
                "Earnings Oracle": 0.10,
            },
            self.HIGH_VOL: {
                "Technical Analyst": 0.20,
                "Flow Detective": 0.20,
                "Regime Sentinel": 0.15,
                "Risk Contrarian": 0.40,  # Highest in high-vol
                "Earnings Oracle": 0.05,
            },
            self.LOW_VOL: {
                "Technical Analyst": 0.30,
                "Flow Detective": 0.25,
                "Regime Sentinel": 0.10,
                "Risk Contrarian": 0.15,
                "Earnings Oracle": 0.20,
            },
        }
        
        if regime is None:
            return weights[self.LOW_VOL]  # Default
        
        return weights.get(regime, weights[self.LOW_VOL])
