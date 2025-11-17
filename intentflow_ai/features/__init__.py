"""Feature engineering interfaces."""

from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.labels import make_excess_label
from intentflow_ai.features.validation import print_validation_report, validate_features

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
        "validate_features",
        "print_validation_report",
    ]
except ImportError:
    __all__ = [
        "FeatureEngineer",
        "make_excess_label",
        "validate_features",
        "print_validation_report",
    ]

# Feature audit tools
try:
    from intentflow_ai.features.audit import (
        FeatureAuditConfig,
        audit_features_by_regime,
        audit_feature_stability,
        identify_feature_clusters,
    )
    __all__.extend([
        "FeatureAuditConfig",
        "audit_features_by_regime",
        "audit_feature_stability",
        "identify_feature_clusters",
    ])
except ImportError:
    pass

# IC diagnostics tools
try:
    from intentflow_ai.features.ic_diagnostics import (
        ICDiagnosticsConfig,
        analyze_cross_feature_correlation,
        analyze_ic_by_feature_block,
        analyze_ic_by_regime,
        analyze_rolling_ic_over_time,
        build_orthogonal_factors,
        detect_ic_breakpoints,
        compute_ic,
        compute_return_ic,
        compute_contribution_ic,
    )
    __all__.extend([
        "ICDiagnosticsConfig",
        "analyze_cross_feature_correlation",
        "analyze_ic_by_feature_block",
        "analyze_ic_by_regime",
        "analyze_rolling_ic_over_time",
        "build_orthogonal_factors",
        "detect_ic_breakpoints",
        "compute_ic",
        "compute_return_ic",
        "compute_contribution_ic",
    ])
except ImportError:
    pass
