"""
Technical Analyst Agent - Agent 1 of Council of Experts.

Wraps the existing V3 LightGBM model for technical analysis.
Focus: Price patterns, momentum, technical indicators.

This agent uses the same hyperparameters as V3:
- n_estimators: 800
- max_depth: 3
- learning_rate: 0.02
- Strong L1/L2 regularization
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from intentflow_ai.agents.base_agent import BaseAgent, AgentOutput
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalAnalystAgent(BaseAgent):
    """
    Agent 1: Technical Analyst
    
    Uses LightGBM to analyze price patterns and momentum.
    This is the SAME model architecture as V3, wrapped in agent interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Technical Analyst", config=config or {})
        
        # V3's hyperparameters
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': self.config.get('n_estimators', 800),
            'max_depth': self.config.get('max_depth', 3),
            'learning_rate': self.config.get('learning_rate', 0.02),
            'num_leaves': self.config.get('num_leaves', 8),
            'min_data_in_leaf': self.config.get('min_data_in_leaf', 800),
            'subsample': self.config.get('subsample', 0.6),
            'feature_fraction': self.config.get('feature_fraction', 0.5),
            'reg_lambda': self.config.get('reg_lambda', 20.0),  # Strong L2
            'reg_alpha': self.config.get('reg_alpha', 5.0),      # Strong L1
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
        }
        
        self._explainer = None
    
    def get_required_features(self) -> List[str]:
        """
        Technical features from V3's feature engineering.
        
        These are the core technical indicators the agent uses.
        """
        return [
            # Technical indicators
            'technical__ema_20',
            'technical__ema_50',
            'technical__macd',
            'technical__macd_signal',
            'technical__boll_z',
            'technical__rsi_14',
            
            # Momentum features
            'momentum__price_ret_1d',
            'momentum__price_ret_3d',
            'momentum__price_ret_5d',
            'momentum__price_ret_10d',
            'momentum__price_ret_20d',
            'momentum__price_mom_5',
            'momentum__price_mom_10',
            'momentum__price_mom_20',
            'momentum__momentum_ratio_10_30',
            'momentum__pct_from_120d_high',
            
            # Volatility
            'volatility__vol_5d',
            'volatility__price_vol_10',
            'volatility__price_vol_20',
            'volatility__downside_vol_10d',
            'volatility__vol_ratio_short_long',
            
            # ATR
            'atr__atr_14',
            'atr__atr_pct_14',
            
            # Volume
            'turnover__volume_mean_20',
            'turnover__volume_spike',
            'turnover__turnover_z_20',
        ]
    
    def _get_available_features(self, X: pd.DataFrame) -> List[str]:
        """Get features that are available in the dataframe."""
        required = self.get_required_features()
        available = [f for f in required if f in X.columns]
        
        if len(available) < len(required):
            missing = set(required) - set(available)
            logger.debug(f"Missing {len(missing)} features: {list(missing)[:5]}...")
        
        return available
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train LightGBM model - same as V3."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        features = self._get_available_features(X_train)
        if len(features) == 0:
            raise ValueError("No valid features found in training data")
        
        self._feature_names = features
        
        logger.info(f"Training {self.name} with {len(features)} features on {len(X_train)} samples")
        
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(X_train[features], y_train)
        self.is_trained = True
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration_}")
    
    def predict(self, X: pd.DataFrame) -> AgentOutput:
        """Predict with SHAP-based explanation."""
        if not self.is_trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        features = X[self._feature_names] if hasattr(self, '_feature_names') else X[self._get_available_features(X)]
        
        # Get raw prediction
        signal_raw = self.model.predict(features)
        if isinstance(signal_raw, np.ndarray):
            signal_raw = signal_raw[0]
        
        # Normalize to [-1, 1] using tanh
        signal = float(np.tanh(signal_raw * 20))  # Scale by 20 since returns are small
        
        # Compute confidence based on prediction magnitude
        confidence = min(abs(signal_raw) / 0.03, 1.0)  # 3% return = max confidence
        
        # Feature attribution via SHAP
        feature_attribution = self._compute_feature_attribution(features)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features, feature_attribution, signal)
        
        return AgentOutput(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            feature_attribution=feature_attribution,
            agent_name=self.name
        )
    
    def _compute_feature_attribution(self, features: pd.DataFrame) -> Dict[str, float]:
        """Compute SHAP values for feature attribution."""
        try:
            import shap
            
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
            
            shap_values = self._explainer.shap_values(features)
            
            if isinstance(shap_values, np.ndarray):
                values = shap_values[0] if shap_values.ndim > 1 else shap_values
            else:
                values = shap_values
            
            return dict(zip(self._feature_names, values.flatten()))
            
        except Exception as e:
            logger.warning(f"SHAP failed: {e}. Using feature importance instead.")
            
            # Fallback to built-in feature importance
            importance = self.model.feature_importances_
            return dict(zip(self._feature_names, importance))
    
    def _generate_reasoning(
        self, 
        features: pd.DataFrame, 
        attribution: Dict[str, float],
        signal: float
    ) -> str:
        """Generate human-readable reasoning."""
        # Get top 3 contributing features
        sorted_attr = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        direction = "bullish" if signal > 0 else "bearish" if signal < 0 else "neutral"
        
        feature_summary = ", ".join([
            f"{name.split('__')[-1]}={features[name].iloc[0]:.3f} (impact: {impact:+.4f})"
            for name, impact in sorted_attr
        ])
        
        return f"Technical analysis is {direction}. Key drivers: {feature_summary}"
