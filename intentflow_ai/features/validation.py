"""Feature validation and sanity checks for IntentFlow AI."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def validate_features(
    feature_frame: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    check_nans: bool = True,
    check_infs: bool = True,
    check_ranges: bool = True,
    sample_tickers: Optional[List[str]] = None,
    sample_dates: Optional[List[pd.Timestamp]] = None,
) -> Dict[str, any]:
    """Validate feature engineering output for common issues.
    
    Args:
        feature_frame: DataFrame with computed features (output of FeatureEngineer.build())
        dataset: Original dataset with date, ticker, and other base columns
        check_nans: Whether to check for unexpected NaN values
        check_infs: Whether to check for infinite values
        check_ranges: Whether to check for unreasonable value ranges
        sample_tickers: Optional list of tickers to show detailed samples for
        sample_dates: Optional list of dates to show detailed samples for
    
    Returns:
        Dictionary with validation results and statistics
    """
    results: Dict[str, any] = {
        "total_features": len(feature_frame.columns),
        "total_rows": len(feature_frame),
        "issues": [],
        "warnings": [],
        "stats": {},
    }
    
    # Align indices if needed
    if len(feature_frame) != len(dataset):
        results["warnings"].append(
            f"Feature frame length ({len(feature_frame)}) != dataset length ({len(dataset)})"
        )
    
    # Check for NaNs
    if check_nans:
        nan_counts = feature_frame.isna().sum()
        nan_pct = (nan_counts / len(feature_frame)) * 100
        high_nan_features = nan_pct[nan_pct > 50].sort_values(ascending=False)
        if len(high_nan_features) > 0:
            results["issues"].append(
                f"Features with >50% NaN: {high_nan_features.to_dict()}"
            )
        results["stats"]["nan_summary"] = {
            "features_with_any_nan": (nan_counts > 0).sum(),
            "features_with_high_nan": (nan_pct > 50).sum(),
            "max_nan_pct": nan_pct.max(),
        }
    
    # Check for infinities
    if check_infs:
        inf_counts = np.isinf(feature_frame.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            inf_features = inf_counts[inf_counts > 0].sort_values(ascending=False)
            results["issues"].append(
                f"Features with infinite values: {inf_features.to_dict()}"
            )
        results["stats"]["inf_summary"] = {
            "features_with_inf": (inf_counts > 0).sum(),
            "total_inf_values": inf_counts.sum(),
        }
    
    # Check value ranges (for numeric features)
    if check_ranges:
        numeric_features = feature_frame.select_dtypes(include=[np.number])
        range_stats = {}
        for col in numeric_features.columns:
            col_data = numeric_features[col].dropna()
            if len(col_data) > 0:
                range_stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                }
                # Flag extreme values
                if abs(col_data.max()) > 1e6 or abs(col_data.min()) > 1e6:
                    results["warnings"].append(
                        f"Feature {col} has extreme values: min={col_data.min():.2e}, max={col_data.max():.2e}"
                    )
        results["stats"]["ranges"] = range_stats
    
    # Sample feature values for inspection
    if sample_tickers or sample_dates:
        sample_mask = pd.Series(False, index=feature_frame.index)
        if sample_tickers and "ticker" in dataset.columns:
            sample_mask |= dataset["ticker"].isin(sample_tickers)
        if sample_dates and "date" in dataset.columns:
            sample_mask |= pd.to_datetime(dataset["date"]).isin(sample_dates)
        
        if sample_mask.any():
            sample_features = feature_frame.loc[sample_mask]
            sample_base = dataset.loc[sample_mask, ["date", "ticker"]].copy() if "date" in dataset.columns and "ticker" in dataset.columns else None
            
            results["samples"] = {
                "n_samples": sample_mask.sum(),
                "features": sample_features.to_dict("records")[:10],  # Limit to 10 rows
            }
            if sample_base is not None:
                results["samples"]["base_info"] = sample_base.to_dict("records")[:10]
    
    # Feature naming consistency check
    feature_names = list(feature_frame.columns)
    expected_prefixes = ["momentum__", "volatility__", "delivery__", "sector_relative__", "regime__", "turnover__"]
    unprefixed = [name for name in feature_names if not any(name.startswith(p) for p in expected_prefixes)]
    if unprefixed:
        results["warnings"].append(
            f"Features without expected prefixes: {unprefixed[:10]}"  # Show first 10
        )
    
    # Summary
    results["summary"] = {
        "passed": len(results["issues"]) == 0,
        "n_issues": len(results["issues"]),
        "n_warnings": len(results["warnings"]),
    }
    
    return results


def print_validation_report(results: Dict[str, any]) -> None:
    """Print a human-readable validation report."""
    print("\n" + "=" * 80)
    print("FEATURE VALIDATION REPORT")
    print("=" * 80)
    
    print(f"\nTotal features: {results['total_features']}")
    print(f"Total rows: {results['total_rows']}")
    
    if results["stats"]:
        print("\nStatistics:")
        if "nan_summary" in results["stats"]:
            nan_sum = results["stats"]["nan_summary"]
            print(f"  Features with NaN: {nan_sum['features_with_any_nan']}")
            print(f"  Features with >50% NaN: {nan_sum['features_with_high_nan']}")
            print(f"  Max NaN percentage: {nan_sum['max_nan_pct']:.1f}%")
        
        if "inf_summary" in results["stats"]:
            inf_sum = results["stats"]["inf_summary"]
            print(f"  Features with Inf: {inf_sum['features_with_inf']}")
            print(f"  Total Inf values: {inf_sum['total_inf_values']}")
    
    if results["issues"]:
        print("\n❌ ISSUES FOUND:")
        for issue in results["issues"]:
            print(f"  - {issue}")
    else:
        print("\n✅ No critical issues found")
    
    if results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    if "samples" in results:
        print(f"\n📊 Sample data ({results['samples']['n_samples']} rows):")
        if "base_info" in results["samples"]:
            print("  Base info (first 3 rows):")
            for i, row in enumerate(results["samples"]["base_info"][:3]):
                print(f"    {i+1}. {row}")
        print("  Feature values (first 3 rows, first 5 features):")
        for i, row in enumerate(results["samples"]["features"][:3]):
            feat_items = list(row.items())[:5]
            print(f"    {i+1}. {dict(feat_items)}")
    
    print("\n" + "=" * 80)
    print(f"Validation {'PASSED' if results['summary']['passed'] else 'FAILED'}")
    print("=" * 80 + "\n")

