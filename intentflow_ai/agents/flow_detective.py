"""
Flow Detective Agent - Agent 2 of Council of Experts.

NEW AGENT - Detects institutional activity from delivery data.
This is IntentFlow V4.5's SECRET WEAPON.

Institutional investors (FII, DII, HNIs) leave footprints in delivery volume
that often precede price moves. The Flow Detective identifies:
- Unusual delivery spikes (accumulation)
- Delivery trend changes
- FII/DII flow patterns

Uses XGBoost which excels at non-linear flow patterns.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from intentflow_ai.agents.base_agent import BaseAgent, AgentOutput
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class FlowDetectiveAgent(BaseAgent):
    """
    Agent 2: Flow Detective
    
    Specializes in detecting institutional buying/selling patterns from delivery data.
    Uses XGBoost for non-linear pattern detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Flow Detective", config=config or {})
        
        # XGBoost params optimized for flow detection
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': self.config.get('max_depth', 4),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'n_estimators': self.config.get('n_estimators', 300),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'reg_lambda': self.config.get('reg_lambda', 1.0),
            'reg_alpha': self.config.get('reg_alpha', 0.5),
            'random_state': 42,
            'verbosity': 0,
        }
    
    def get_required_features(self) -> List[str]:
        """
        Delivery-focused features from delivery_features.py.
        Also includes FII/DII institutional flow data.
        """
        return [
            # Delivery momentum (from delivery_features.py)
            'deliv_pct_3d_change',
            'deliv_pct_7d_change',
            'deliv_pct_14d_change',
            
            # Delivery spikes
            'deliv_qty_zscore',
            'large_delivery_flag',
            
            # Accumulation/Distribution
            'accum_distrib_5d',
            'accum_distrib_10d',
            
            # Delivery ratios
            'deliv_ratio_ma_5',
            'deliv_ratio_ma_20',
            'deliv_ratio_trend',
            
            # Delivery value
            'deliv_value_spike',
            
            # Price-delivery correlation
            'deliv_price_corr_10',
            
            # FII/DII institutional flow (if available)
            'fii_net_5d',
            'dii_net_5d',
            'fii_dii_ratio',
        ]
    
    def _get_available_features(self, X: pd.DataFrame) -> List[str]:
        """Get features that are available in the dataframe."""
        required = self.get_required_features()
        available = [f for f in required if f in X.columns]
        
        # Log missing critical features
        missing = set(required) - set(available)
        if 'deliv_qty_zscore' in missing:
            logger.warning("Critical delivery feature 'deliv_qty_zscore' missing - agent may underperform")
        
        return available
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model on delivery features."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        features = self._get_available_features(X_train)
        
        if len(features) == 0:
            logger.warning(f"No delivery features found. Creating placeholder model.")
            self.model = None
            self.is_trained = True
            self._feature_names = []
            return
        
        self._feature_names = features
        
        logger.info(f"Training {self.name} with {len(features)} features on {len(X_train)} samples")
        
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_train[features], y_train)
        self.is_trained = True
        
        logger.info(f"Training complete.")
    
    def predict(self, X: pd.DataFrame) -> AgentOutput:
        """Predict with delivery-focused reasoning."""
        if not self.is_trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        # Handle case where no delivery features available
        if self.model is None or len(self._feature_names) == 0:
            return AgentOutput(
                signal=0.0,
                confidence=0.0,
                reasoning="No delivery data available - agent inactive",
                feature_attribution={},
                agent_name=self.name
            )
        
        features = X[self._feature_names]
        
        # Get raw prediction
        signal_raw = self.model.predict(features)
        if isinstance(signal_raw, np.ndarray):
            signal_raw = signal_raw[0]
        
        # Normalize to [-1, 1]
        signal = float(np.tanh(signal_raw * 25))
        
        # Confidence boosted by delivery spike detection
        delivery_spike = self._detect_delivery_spike(X)
        base_confidence = min(abs(signal_raw) / 0.025, 1.0)
        confidence = min(base_confidence * (1.5 if delivery_spike else 1.0), 1.0)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(X, signal, delivery_spike)
        
        # Feature attribution
        feature_attribution = self._compute_feature_attribution(features)
        
        return AgentOutput(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            feature_attribution=feature_attribution,
            agent_name=self.name
        )
    
    def _detect_delivery_spike(self, X: pd.DataFrame) -> bool:
        """Check if there's an unusual delivery spike."""
        if 'large_delivery_flag' in X.columns:
            return bool(X['large_delivery_flag'].iloc[0] == 1)
        if 'deliv_qty_zscore' in X.columns:
            return bool(X['deliv_qty_zscore'].iloc[0] > 2.0)
        return False
    
    def _generate_reasoning(self, X: pd.DataFrame, signal: float, spike: bool) -> str:
        """Generate flow-focused reasoning."""
        parts = []
        
        # Spike detection
        if spike:
            zscore = X.get('deliv_qty_zscore', pd.Series([0])).iloc[0]
            parts.append(f"🚨 Large institutional delivery detected ({zscore:.1f}σ above normal)")
        
        # Delivery trend
        if 'deliv_pct_7d_change' in X.columns:
            change_7d = X['deliv_pct_7d_change'].iloc[0]
            if abs(change_7d) > 0.1:
                direction = "increasing" if change_7d > 0 else "decreasing"
                parts.append(f"Delivery trend {direction} ({change_7d:+.1%} over 7d)")
        
        # Accumulation/Distribution
        if 'accum_distrib_5d' in X.columns:
            ad = X['accum_distrib_5d'].iloc[0]
            if abs(ad) > 1e6:  # Significant flow
                flow = "accumulation" if ad > 0 else "distribution"
                parts.append(f"Money flow shows {flow}")
        
        # FII/DII info if available
        if 'fii_net_5d' in X.columns and 'dii_net_5d' in X.columns:
            fii = X['fii_net_5d'].iloc[0]
            dii = X['dii_net_5d'].iloc[0]
            if abs(fii) > 100 or abs(dii) > 100:  # In crores
                parts.append(f"FII: ₹{fii:.0f}Cr, DII: ₹{dii:.0f}Cr (5d net)")
        
        if not parts:
            direction = "bullish" if signal > 0 else "bearish" if signal < 0 else "neutral"
            parts.append(f"Flow analysis suggests {direction} positioning")
        
        return " | ".join(parts)
    
    def _compute_feature_attribution(self, features: pd.DataFrame) -> Dict[str, float]:
        """Compute feature importance."""
        if self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importances_
            return dict(zip(self._feature_names, importance))
        except Exception:
            return {}
