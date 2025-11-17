"""Feature engineering interfaces."""

from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.labels import make_excess_label

try:
    from intentflow_ai.features.orthogonality import (
        FeatureOrthogonalityAnalyzer,
        OrthogonalityConfig,
        generate_orthogonality_report,
        test_incremental_ic,
    )
    __all__ = [
        "FeatureEngineer",
        "make_excess_label",
        "FeatureOrthogonalityAnalyzer",
        "OrthogonalityConfig",
        "generate_orthogonality_report",
        "test_incremental_ic",
    ]
except ImportError:
    __all__ = ["FeatureEngineer", "make_excess_label"]
