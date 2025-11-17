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

# Trading metrics (new comprehensive metrics)
try:
    from intentflow_ai.modeling.trading_metrics import (
        TradingMetrics,
        compare_in_sample_vs_out_of_sample,
        compute_comprehensive_trading_metrics,
        compute_contribution_ic,
        compute_decile_ic,
        compute_expected_value,
        compute_hit_rate as compute_hit_rate_trading,
        compute_profit_factor,
        compute_return_ic,
        compute_sharpe_by_decile,
    )
    _trading_metrics_available = True
except ImportError:
    _trading_metrics_available = False

try:
    from intentflow_ai.modeling.explanations import (
        ExplanationConfig,
        explain_signals,
        SHAPExplainer,
    )
    _explanations_available = True
except ImportError:
    _explanations_available = False

# Build __all__ list
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

if _trading_metrics_available:
    __all__.extend([
        "TradingMetrics",
        "compute_comprehensive_trading_metrics",
        "compute_expected_value",
        "compute_profit_factor",
        "compute_hit_rate_trading",
        "compute_return_ic",
        "compute_contribution_ic",
        "compute_decile_ic",
        "compute_sharpe_by_decile",
        "compare_in_sample_vs_out_of_sample",
    ])

if _explanations_available:
    __all__.extend([
        "ExplanationConfig",
        "explain_signals",
        "SHAPExplainer",
    ])
