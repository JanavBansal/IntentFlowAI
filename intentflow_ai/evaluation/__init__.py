"""Evaluation and benchmarking modules."""

from intentflow_ai.evaluation.benchmarks import (
    BenchmarkComparison,
    BenchmarkConfig,
    BenchmarkResult,
    compute_benchmark_returns,
    generate_comparison_report,
)

from intentflow_ai.evaluation.attribution import (
    PerformanceAttributor,
    AttributionConfig,
    AttributionReport,
    FactorReturn,
    SectorAttribution,
    generate_attribution_summary,
)

__all__ = [
    # Benchmarks
    "BenchmarkComparison",
    "BenchmarkConfig",
    "BenchmarkResult",
    "compute_benchmark_returns",
    "generate_comparison_report",
    # Attribution
    "PerformanceAttributor",
    "AttributionConfig",
    "AttributionReport",
    "FactorReturn",
    "SectorAttribution",
    "generate_attribution_summary",
]
