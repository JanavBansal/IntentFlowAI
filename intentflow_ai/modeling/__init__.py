"""Model training, evaluation, and regime logic."""

from intentflow_ai.modeling.metrics import hit_rate, precision_at_k
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.modeling.evaluation import ModelEvaluator
from intentflow_ai.modeling.regimes import RegimeClassifier

try:
    from intentflow_ai.modeling.explanations import (
        ExplanationConfig,
        explain_signals,
        SHAPExplainer,
    )
    __all__ = [
        "LightGBMTrainer",
        "ModelEvaluator",
        "RegimeClassifier",
        "ExplanationConfig",
        "explain_signals",
        "SHAPExplainer",
        "precision_at_k",
        "hit_rate",
    ]
except ImportError:
    __all__ = [
        "LightGBMTrainer",
        "ModelEvaluator",
        "RegimeClassifier",
        "precision_at_k",
        "hit_rate",
    ]
