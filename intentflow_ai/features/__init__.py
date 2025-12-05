"""Feature engineering interfaces."""

from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.labels import make_excess_label, make_triple_barrier_label
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
        "make_triple_barrier_label",
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
        "make_triple_barrier_label",
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

# Quality scores (Piotroski F-Score, Altman Z-Score)
try:
    from intentflow_ai.features.quality_scores import (
        piotroski_f_score,
        altman_z_score,
        quality_composite_score,
        get_quality_score,
        compute_quality_features,
    )
    __all__.extend([
        "piotroski_f_score",
        "altman_z_score",
        "quality_composite_score",
        "get_quality_score",
        "compute_quality_features",
    ])
except ImportError:
    pass

# Options features
try:
    from intentflow_ai.features.options_features import (
        compute_options_features,
        is_fno_stock,
        add_options_features_to_df,
        get_pcr_signal,
    )
    __all__.extend([
        "compute_options_features",
        "is_fno_stock",
        "add_options_features_to_df",
        "get_pcr_signal",
    ])
except ImportError:
    pass

# Seasonality features
try:
    from intentflow_ai.features.seasonality import (
        get_seasonality_features,
        get_sector_seasonality,
        SeasonalityConfig,
    )
    __all__.extend([
        "get_seasonality_features",
        "get_sector_seasonality",
        "SeasonalityConfig",
    ])
except ImportError:
    pass

# Advanced features integration
try:
    from intentflow_ai.features.advanced_features import (
        build_all_advanced_features,
        get_advanced_feature_blocks,
        build_quality_score_features,
        build_options_features,
        build_macro_features,
        build_seasonality_features,
        build_market_cap_features,
    )
    __all__.extend([
        "build_all_advanced_features",
        "get_advanced_feature_blocks",
        "build_quality_score_features",
        "build_options_features",
        "build_macro_features",
        "build_seasonality_features",
        "build_market_cap_features",
    ])
except ImportError:
    pass
