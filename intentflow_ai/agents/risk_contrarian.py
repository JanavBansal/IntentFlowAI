"""
Risk Contrarian Agent - Agent 4 of Council of Experts.

The Skeptic - has VETO power over other agents.

This agent detects market traps, anomalies, and high-risk conditions:
- Liquidity crashes
- Correlation breakdowns
- Unusual volatility spikes
- Bull/bear traps

Uses Isolation Forest for anomaly detection and Random Forest for trap classification.
When trap probability > 70%, this agent can VETO the entire trade.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from intentflow_ai.agents.base_agent import BaseAgent, AgentOutput
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class RiskContrarianAgent(BaseAgent):
    """
    Agent 4: Risk Contrarian
    
    The guardian of the portfolio - detects traps and anomalies.
    Has VETO power: if trap probability > threshold, trade is cancelled.
    
    Two models:
    1. Isolation Forest: Unsupervised anomaly detection
    2. Random Forest: Supervised trap classification (trained on historical drawdowns)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Risk Contrarian", config=config or {})
        
        self.veto_threshold = self.config.get('veto_threshold', 0.70)
        self.anomaly_contamination = self.config.get('anomaly_contamination', 0.10)
        
        self._anomaly_detector = None
        self._trap_classifier = None
    
    def get_required_features(self) -> List[str]:
        """Features for risk/trap detection."""
        return [
            # Volatility signals
            'volatility__vol_ratio_short_long',  # Sudden vol spikes
            'volatility__price_vol_20',
            
            # ATR for range expansion
            'atr__atr_pct_14',
            
            # Volume anomalies
            'turnover__volume_spike',
            'turnover__turnover_z_20',
            
            # Technical extremes
            'technical__rsi_14',
            'technical__boll_z',
            
            # Momentum reversals
            'momentum__price_ret_1d',
            'momentum__price_ret_5d',
            
            # Market-wide indicators (if available)
            'india_vix',
            'market_breadth',  # Advance-decline
        ]
    
    def _get_available_features(self, X: pd.DataFrame) -> List[str]:
        """Get features that exist in the dataframe."""
        return [f for f in self.get_required_features() if f in X.columns]
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train risk detection models.
        
        Args:
            X_train: Features
            y_train: Binary labels (1 = trap/loss > 3% in next period, 0 = normal)
        """
        from sklearn.ensemble import IsolationForest, RandomForestClassifier
        
        features = self._get_available_features(X_train)
        
        if len(features) == 0:
            logger.warning("No risk features available. Creating placeholder models.")
            self.is_trained = True
            self._feature_names = []
            return
        
        self._feature_names = features
        X_clean = X_train[features].fillna(0).replace([np.inf, -np.inf], 0)
        
        logger.info(f"Training {self.name} with {len(features)} features")
        
        # 1. Anomaly detector (unsupervised)
        self._anomaly_detector = IsolationForest(
            contamination=self.anomaly_contamination,
            random_state=42,
            n_jobs=-1
        )
        self._anomaly_detector.fit(X_clean)
        
        # 2. Trap classifier (supervised)
        # Convert y to binary: 1 if significant loss, 0 otherwise
        y_binary = (y_train < -0.03).astype(int)  # 3% loss = trap
        
        self._trap_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=100,
            random_state=42,
            n_jobs=-1
        )
        self._trap_classifier.fit(X_clean, y_binary)
        
        self.is_trained = True
        logger.info("Risk models training complete")
    
    def predict(self, X: pd.DataFrame) -> AgentOutput:
        """Evaluate risk and potentially veto."""
        if not self.is_trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        if len(self._feature_names) == 0:
            return AgentOutput(
                signal=0.0,
                confidence=0.0,
                reasoning="No risk features available - agent inactive",
                feature_attribution={},
                agent_name=self.name
            )
        
        features = X[self._feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 1. Anomaly detection
        is_anomaly = False
        anomaly_score = 0.0
        if self._anomaly_detector is not None:
            anomaly_pred = self._anomaly_detector.predict(features)
            is_anomaly = anomaly_pred[0] == -1
            # Get anomaly score (more negative = more anomalous)
            anomaly_score = -self._anomaly_detector.score_samples(features)[0]
        
        # 2. Trap probability
        trap_prob = 0.0
        if self._trap_classifier is not None:
            trap_prob = self._trap_classifier.predict_proba(features)[0][1]
        
        # 3. Determine signal and veto
        should_veto = (trap_prob > self.veto_threshold) or is_anomaly
        
        if should_veto:
            signal = -1.0  # Strong negative = veto
            confidence = max(trap_prob, 0.8 if is_anomaly else 0.0)
        else:
            signal = 0.0  # Neutral - no objection
            confidence = 1.0 - trap_prob  # Confidence in safety
        
        # Generate reasoning
        reasoning = self._generate_reasoning(X, is_anomaly, trap_prob, should_veto)
        
        # Feature attribution
        feature_attr = self._compute_feature_attribution(features)
        
        return AgentOutput(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            feature_attribution=feature_attr,
            agent_name=self.name
        )
    
    def _generate_reasoning(
        self,
        X: pd.DataFrame,
        is_anomaly: bool,
        trap_prob: float,
        should_veto: bool
    ) -> str:
        """Generate risk-focused reasoning."""
        parts = []
        
        if should_veto:
            parts.append(f"🛑 VETO: High risk detected (trap probability: {trap_prob:.0%})")
            if is_anomaly:
                parts.append("⚠️ Market structure anomaly detected")
        else:
            parts.append(f"Risk acceptable (trap probability: {trap_prob:.0%})")
        
        # Add specific risk factors
        if 'india_vix' in X.columns:
            vix = X['india_vix'].iloc[0]
            if vix > 25:
                parts.append(f"Elevated VIX: {vix:.1f}")
        
        if 'volatility__vol_ratio_short_long' in X.columns:
            vol_ratio = X['volatility__vol_ratio_short_long'].iloc[0]
            if vol_ratio > 1.5:
                parts.append(f"Vol spike: {vol_ratio:.1f}x above normal")
        
        if 'turnover__volume_spike' in X.columns:
            vol_spike = X['turnover__volume_spike'].iloc[0]
            if vol_spike > 2.0:
                parts.append(f"Unusual volume: {vol_spike:.1f}x")
        
        return " | ".join(parts)
    
    def _compute_feature_attribution(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance from trap classifier."""
        if self._trap_classifier is None:
            return {}
        
        try:
            importance = self._trap_classifier.feature_importances_
            return dict(zip(self._feature_names, importance))
        except Exception:
            return {}
    
    def should_veto(self, output: AgentOutput) -> bool:
        """Check if this agent's output indicates a veto."""
        return output.signal < -0.5 and output.confidence > 0.7
