"""Model training, evaluation, and regime logic."""

from intentflow_ai.modeling.metrics import hit_rate, precision_at_k
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.modeling.evaluation import ModelEvaluator
from intentflow_ai.modeling.regimes import (
    RegimeClassifier,
    RegimeConfig,
    apply_regime_filter_to_signals,
)
from intentflow_ai.modeling.signal_cards import SignalCard, SignalCardGenerator
from intentflow_ai.modeling.stability import (
    StabilityOptimizer,
    StabilityConfig,
    compare_to_baseline,
    generate_stability_report,
)

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
        "RegimeConfig",
        "apply_regime_filter_to_signals",
        "SignalCard",
        "SignalCardGenerator",
        "StabilityOptimizer",
        "StabilityConfig",
        "compare_to_baseline",
        "generate_stability_report",
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
        "RegimeConfig",
        "apply_regime_filter_to_signals",
        "SignalCard",
        "SignalCardGenerator",
        "StabilityOptimizer",
        "StabilityConfig",
        "compare_to_baseline",
        "generate_stability_report",
        "precision_at_k",
        "hit_rate",
    ]
