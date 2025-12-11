"""
Base Agent Interface for IntentFlow V4.5 Council of Experts.

All agents must implement this interface to participate in the Council.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AgentOutput:
    """Standardized output from any agent in the Council."""
    
    # Core signal
    signal: float  # -1 to 1 (sell to buy)
    confidence: float  # 0 to 1
    
    # Explainability
    reasoning: str  # Natural language explanation
    feature_attribution: Optional[Dict[str, float]] = None  # SHAP values or feature importance
    
    # Metadata
    agent_name: str = ""
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Validate signal and confidence are in expected ranges."""
        self.signal = max(-1.0, min(1.0, self.signal))  # Clamp to [-1, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))  # Clamp to [0, 1]
    
    @property
    def direction(self) -> str:
        """Human-readable signal direction."""
        if self.signal > 0.1:
            return "BULLISH"
        elif self.signal < -0.1:
            return "BEARISH"
        return "NEUTRAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signal": self.signal,
            "confidence": self.confidence,
            "direction": self.direction,
            "reasoning": self.reasoning,
            "feature_attribution": self.feature_attribution,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentOutput":
        """Create AgentOutput from dictionary, filtering out computed properties."""
        # Filter to only constructor args
        valid_keys = {'signal', 'confidence', 'reasoning', 'feature_attribution', 'agent_name', 'timestamp'}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Council.
    
    Each agent specializes in a specific aspect of market analysis:
    - Technical: Price patterns and momentum
    - Flow: Institutional activity and delivery data
    - Regime: Market state detection
    - Risk: Anomaly detection and trap avoidance
    - Earnings: Fundamental screening
    
    Agents must implement:
    - train(): Train the underlying model
    - predict(): Generate signal with explanation
    - get_required_features(): List features this agent needs
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            name: Human-readable agent name
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self._feature_names: List[str] = []
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the agent's model.
        
        Args:
            X_train: Training features (must contain columns from get_required_features())
            y_train: Target variable (usually forward returns)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> AgentOutput:
        """
        Generate signal and explanation for given features.
        
        Args:
            X: Feature DataFrame (single row or multiple rows)
            
        Returns:
            AgentOutput with signal, confidence, and reasoning
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Return list of feature names this agent needs.
        
        Used by the Council to ensure all required features are available
        before calling predict().
        """
        pass
    
    def validate_features(self, X: pd.DataFrame) -> bool:
        """Check if all required features are present."""
        required = set(self.get_required_features())
        available = set(X.columns)
        missing = required - available
        
        if missing:
            from intentflow_ai.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"{self.name}: Missing features: {missing}")
            return False
        return True
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        import joblib
        if self.model is not None:
            joblib.dump({
                "model": self.model,
                "name": self.name,
                "config": self.config,
                "is_trained": self.is_trained,
                "feature_names": self._feature_names,
            }, path)
    
    @classmethod
    def load(cls, path: str) -> "BaseAgent":
        """Load trained model from disk."""
        import joblib
        data = joblib.load(path)
        
        agent = cls(name=data["name"], config=data["config"])
        agent.model = data["model"]
        agent.is_trained = data["is_trained"]
        agent._feature_names = data.get("feature_names", [])
        
        return agent
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"
