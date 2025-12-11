"""
Earnings Oracle Agent - Agent 5 of Council of Experts.

Fundamental screening agent focused on:
- Valuation (P/E, P/B)
- Profitability (ROE, ROCE)
- Earnings quality and surprises
- Earnings event proximity

Uses simple Logistic Regression for interpretability.
Signal is weak (max ±0.15) because fundamentals don't time trades well,
but they help avoid value traps.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from intentflow_ai.agents.base_agent import BaseAgent, AgentOutput
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class EarningsOracleAgent(BaseAgent):
    """
    Agent 5: Earnings Oracle
    
    Screens for fundamental quality to avoid value traps.
    Uses Logistic Regression for interpretability.
    
    This agent has intentionally weak signals (max ±0.15) because:
    1. Fundamentals change slowly
    2. They don't time entries well
    3. Their value is in filtering out bad stocks, not picking winners
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Earnings Oracle", config=config or {})
        
        # Maximum signal strength (keep weak)
        self.max_signal = self.config.get('max_signal', 0.15)
        
        # Logistic regression params
        self.lr_params = {
            'C': self.config.get('C', 1.0),
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42,
        }
    
    def get_required_features(self) -> List[str]:
        """
        Fundamental features from EODHD data.
        
        These map to columns in the EODHD parquet cache:
        - Valuation: pe_ratio, pb_ratio, etc.
        - Profitability: roe, roa, margins
        - Quality: accruals_ratio, cf_to_net_income
        - Leverage: debt_to_equity, current_ratio
        """
        return [
            # Valuation (from EODHD)
            'pe_ratio',
            'pb_ratio',
            'ev_ebitda',
            'dividend_yield',
            
            # Profitability (from EODHD)
            'roe',
            'roa',
            'gross_margin',
            'operating_margin',
            'net_margin',
            
            # Quality (from EODHD)
            'accruals_ratio',
            'cf_to_net_income',
            
            # Leverage (from EODHD)
            'debt_to_equity',
            'current_ratio',
            
            # Sector-relative (computed in fundamental_features.py)
            'pe_ratio_sector_rel',
            'roe_sector_pct',
        ]
    
    def _get_available_features(self, X: pd.DataFrame) -> List[str]:
        """Get features that exist in the dataframe."""
        return [f for f in self.get_required_features() if f in X.columns]
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train logistic regression on fundamental features.
        
        Args:
            X_train: Fundamental features
            y_train: Forward returns (converted to binary: >0 = 1, <=0 = 0)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        features = self._get_available_features(X_train)
        
        if len(features) == 0:
            logger.warning("No fundamental features available. Creating placeholder model.")
            self.is_trained = True
            self._feature_names = []
            return
        
        self._feature_names = features
        X_clean = X_train[features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_clean)
        
        # Convert y to binary for classification
        y_binary = (y_train > 0).astype(int)
        
        logger.info(f"Training {self.name} with {len(features)} features")
        
        self.model = LogisticRegression(**self.lr_params)
        self.model.fit(X_scaled, y_binary)
        
        self.is_trained = True
        logger.info("Earnings Oracle training complete")
    
    def predict(self, X: pd.DataFrame) -> AgentOutput:
        """Evaluate fundamental quality."""
        if not self.is_trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        if len(self._feature_names) == 0:
            return AgentOutput(
                signal=0.0,
                confidence=0.0,
                reasoning="No fundamental data available - agent inactive",
                feature_attribution={},
                agent_name=self.name
            )
        
        features = X[self._feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        features_scaled = self._scaler.transform(features)
        
        # Get probability of positive return
        quality_prob = self.model.predict_proba(features_scaled)[0][1]
        
        # Convert to weak signal [-0.15, 0.15]
        signal = (quality_prob - 0.5) * 2 * self.max_signal
        
        # Confidence is distance from 50%
        confidence = abs(quality_prob - 0.5) * 2
        
        # Generate reasoning
        reasoning = self._generate_reasoning(X, quality_prob)
        
        # Feature attribution from coefficients
        feature_attr = self._compute_feature_attribution()
        
        return AgentOutput(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            feature_attribution=feature_attr,
            agent_name=self.name
        )
    
    def _generate_reasoning(self, X: pd.DataFrame, quality_prob: float) -> str:
        """Generate fundamental-focused reasoning."""
        parts = [f"Fundamental quality score: {quality_prob:.0%}"]
        
        # Valuation comment
        if 'pe_ratio' in X.columns:
            pe = X['pe_ratio'].iloc[0]
            if pe > 0:
                if pe < 15:
                    parts.append(f"Attractive P/E: {pe:.1f}")
                elif pe > 40:
                    parts.append(f"⚠️ Expensive P/E: {pe:.1f}")
        
        # ROE comment
        if 'roe' in X.columns:
            roe = X['roe'].iloc[0]
            if roe > 0.20:
                parts.append(f"Strong ROE: {roe:.0%}")
            elif roe < 0.05:
                parts.append(f"⚠️ Weak ROE: {roe:.0%}")
        
        # Debt warning
        if 'debt_to_equity' in X.columns:
            de = X['debt_to_equity'].iloc[0]
            if de > 1.5:
                parts.append(f"⚠️ High debt: D/E={de:.1f}")
        
        # Earnings proximity warning
        if 'days_to_earnings' in X.columns:
            days = X['days_to_earnings'].iloc[0]
            if 0 < days < 7:
                parts.append(f"📅 Earnings in {int(days)} days")
        
        # Earnings surprise
        if 'earnings_surprise_pct' in X.columns:
            surprise = X['earnings_surprise_pct'].iloc[0]
            if abs(surprise) > 0.05:
                direction = "beat" if surprise > 0 else "miss"
                parts.append(f"Last quarter {direction}: {surprise:+.0%}")
        
        return " | ".join(parts)
    
    def _compute_feature_attribution(self) -> Dict[str, float]:
        """Get feature importance from logistic regression coefficients."""
        if self.model is None or len(self._feature_names) == 0:
            return {}
        
        try:
            coefs = self.model.coef_[0]
            return dict(zip(self._feature_names, coefs))
        except Exception:
            return {}
