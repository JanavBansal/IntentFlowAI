"""Information Coefficient (IC) diagnostic tools for deep feature analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.backtest.filters import RiskFilterConfig, compute_regime_flags
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ICDiagnosticsConfig:
    """Configuration for IC diagnostics."""

    regime_vol_buckets: List[float] = None
    risk_cfg: RiskFilterConfig = None
    rolling_window_days: int = 252
    min_samples_per_window: int = 50
    min_samples_per_regime: int = 100
    correlation_threshold: float = 0.8

    def __post_init__(self):
        if self.regime_vol_buckets is None:
            self.regime_vol_buckets = [0, 33, 67, 100]
        if self.risk_cfg is None:
            self.risk_cfg = RiskFilterConfig()


def compute_ic(
    feature_values: pd.Series,
    target_values: pd.Series,
    method: str = "pearson",
) -> float:
    """Compute Information Coefficient (correlation).

    Args:
        feature_values: Feature values
        target_values: Target labels or returns
        method: 'pearson' or 'spearman'

    Returns:
        IC value (float)
    """
    aligned = pd.DataFrame({"feature": feature_values, "target": target_values}).dropna()
    if len(aligned) < 10:
        return np.nan
    try:
        return float(aligned["feature"].corr(aligned["target"], method=method))
    except Exception:
        return np.nan


def compute_return_ic(
    feature_values: pd.Series,
    target_values: pd.Series,
) -> float:
    """Compute Return IC (Pearson correlation of signal magnitude with returns).

    This is more intuitive than Rank IC for classification models.

    Args:
        feature_values: Feature values (signal strength)
        target_values: Target returns

    Returns:
        Return IC (Pearson correlation)
    """
    return compute_ic(feature_values, target_values, method="pearson")


def compute_contribution_ic(
    feature_values: pd.Series,
    position_sizes: pd.Series,
    target_values: pd.Series,
) -> float:
    """Compute Contribution-weighted IC.

    Accounts for position sizing - correlation of (signal × position_size) with returns.

    Args:
        feature_values: Feature values (signal strength)
        position_sizes: Position sizes (capital allocation)
        target_values: Target returns

    Returns:
        Contribution IC
    """
    aligned = pd.DataFrame(
        {
            "feature": feature_values,
            "position_size": position_sizes,
            "target": target_values,
        }
    ).dropna()

    if len(aligned) < 10:
        return np.nan

    contribution = aligned["feature"] * aligned["position_size"]

    try:
        return float(contribution.corr(aligned["target"], method="pearson"))
    except Exception:
        return np.nan


def analyze_ic_by_feature_block(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    feature_block_prefixes: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Analyze IC separately for each feature block (data layer).

    Args:
        features: DataFrame with feature columns (prefixed with block name, e.g., "technical__")
        labels: Series with target labels
        feature_block_prefixes: Optional mapping of block names to prefixes.
            If None, will infer from feature column names (format: "block__feature")

    Returns:
        DataFrame with IC statistics by feature block
    """
    if features.empty or labels.empty:
        return pd.DataFrame()

    # Infer feature blocks from column names
    if feature_block_prefixes is None:
        feature_block_prefixes = {}
        for col in features.columns:
            if "__" in col:
                block_name = col.split("__")[0]
                if block_name not in feature_block_prefixes:
                    feature_block_prefixes[block_name] = []
                feature_block_prefixes[block_name].append(col)
            else:
                # Features without block prefix go to "baseline"
                if "baseline" not in feature_block_prefixes:
                    feature_block_prefixes["baseline"] = []
                feature_block_prefixes["baseline"].append(col)

    results = []

    for block_name, feature_cols in feature_block_prefixes.items():
        block_features = features[feature_cols]
        block_ics = []
        block_rank_ics = []
        feature_names = []

        for feat_col in feature_cols:
            if feat_col not in features.columns:
                continue

            feat_values = features[feat_col].dropna()
            aligned_labels = labels.reindex(feat_values.index).dropna()
            feat_values = feat_values.reindex(aligned_labels.index)

            if len(feat_values) < 10:
                continue

            # Return IC (Pearson - signal magnitude correlation)
            return_ic = compute_return_ic(feat_values, aligned_labels)
            # Rank IC (Spearman - rank correlation)
            rank_ic = compute_ic(feat_values, aligned_labels, method="spearman")

            if not pd.isna(return_ic):
                block_ics.append(return_ic)
                block_rank_ics.append(rank_ic)
                feature_names.append(feat_col)

        if block_ics:
            results.append(
                {
                    "feature_block": block_name,
                    "feature_count": len(block_ics),
                    "mean_ic": float(np.nanmean(block_ics)),
                    "median_ic": float(np.nanmedian(block_ics)),
                    "std_ic": float(np.nanstd(block_ics)),
                    "mean_rank_ic": float(np.nanmean(block_rank_ics)),
                    "median_rank_ic": float(np.nanmedian(block_rank_ics)),
                    "std_rank_ic": float(np.nanstd(block_rank_ics)),
                    "positive_ic_count": int(sum(1 for ic in block_ics if ic > 0)),
                    "ic_above_0_1_count": int(sum(1 for ic in block_ics if ic > 0.1)),
                    "top_feature": feature_names[np.argmax(block_ics)] if block_ics else None,
                    "top_ic": float(np.nanmax(block_ics)) if block_ics else np.nan,
                }
            )

    return pd.DataFrame(results).sort_values("mean_ic", ascending=False)


def analyze_rolling_ic_over_time(
    features: pd.DataFrame,
    labels: pd.Series,
    dates: pd.Series,
    *,
    window_days: int = 252,
    min_samples: int = 50,
    step_days: int = 21,  # Step size for rolling windows (monthly)
) -> pd.DataFrame:
    """Compute rolling IC over time to detect feature degradation and breakpoints.

    Args:
        features: DataFrame with feature columns
        labels: Series with target labels
        dates: Series with dates (aligned with features index)
        window_days: Rolling window size in days
        min_samples: Minimum samples per window
        step_days: Step size between windows (default: monthly)

    Returns:
        DataFrame with rolling IC statistics
    """
    if features.empty or labels.empty:
        return pd.DataFrame()

    # Align all series
    aligned = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "label": labels,
        },
        index=features.index,
    )
    aligned = pd.concat([aligned, features], axis=1).dropna(subset=["date", "label"])

    if aligned.empty:
        return pd.DataFrame()

    aligned = aligned.sort_values("date")
    unique_dates = aligned["date"].unique()
    window_size = pd.Timedelta(days=window_days)
    step_size = pd.Timedelta(days=step_days)

    results = []

    # Rolling window analysis
    window_start = unique_dates[0]
    window_end = window_start + window_size

    while window_end <= unique_dates[-1]:
        window_data = aligned[(aligned["date"] >= window_start) & (aligned["date"] < window_end)]

        if len(window_data) < min_samples:
            window_start += step_size
            window_end += step_size
            continue

        window_labels = window_data["label"]
        window_ics = []
        window_rank_ics = []
        feature_cols = [col for col in features.columns if col not in ["date", "ticker"]]

        # Compute aggregate IC for all features in this window
        for feat_col in feature_cols:
            if feat_col not in window_data.columns:
                continue

            feat_values = window_data[feat_col].dropna()
            aligned_labels = window_labels.reindex(feat_values.index).dropna()
            feat_values = feat_values.reindex(aligned_labels.index)

            if len(feat_values) < min_samples:
                continue

            try:
                # Use Return IC (Pearson) for rolling analysis
                return_ic = compute_return_ic(feat_values, aligned_labels)
                rank_ic = compute_ic(feat_values, aligned_labels, method="spearman")
                if not pd.isna(return_ic):
                    window_ics.append(return_ic)
                if not pd.isna(rank_ic):
                    window_rank_ics.append(rank_ic)
            except Exception:
                pass

        if window_ics:
            results.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_center": window_start + (window_end - window_start) / 2,
                    "sample_count": len(window_data),
                    "feature_count": len(window_ics),
                    "mean_ic": float(np.nanmean(window_ics)),
                    "median_ic": float(np.nanmedian(window_ics)),
                    "std_ic": float(np.nanstd(window_ics)),
                    "mean_rank_ic": float(np.nanmean(window_rank_ics)),
                    "median_rank_ic": float(np.nanmedian(window_rank_ics)),
                    "positive_ic_ratio": float(sum(1 for ic in window_ics if ic > 0) / len(window_ics)),
                }
            )

        window_start += step_size
        window_end += step_size

    return pd.DataFrame(results)


def detect_ic_breakpoints(
    rolling_ic: pd.DataFrame,
    *,
    ic_drop_threshold: float = 0.05,
    min_windows_before: int = 3,
    min_windows_after: int = 3,
) -> pd.DataFrame:
    """Detect breakpoints where IC drops significantly (market structure changes).

    Args:
        rolling_ic: DataFrame from analyze_rolling_ic_over_time
        ic_drop_threshold: Minimum IC drop to consider a breakpoint
        min_windows_before: Minimum windows before breakpoint to validate
        min_windows_after: Minimum windows after breakpoint to validate

    Returns:
        DataFrame with detected breakpoints
    """
    if rolling_ic.empty or len(rolling_ic) < min_windows_before + min_windows_after:
        return pd.DataFrame()

    rolling_ic = rolling_ic.sort_values("window_center")
    breakpoints = []

    for i in range(min_windows_before, len(rolling_ic) - min_windows_after):
        before_mean = rolling_ic.iloc[i - min_windows_before : i]["mean_ic"].mean()
        after_mean = rolling_ic.iloc[i : i + min_windows_after]["mean_ic"].mean()
        ic_drop = before_mean - after_mean

        if ic_drop >= ic_drop_threshold:
            breakpoints.append(
                {
                    "breakpoint_date": rolling_ic.iloc[i]["window_center"],
                    "ic_before": float(before_mean),
                    "ic_after": float(after_mean),
                    "ic_drop": float(ic_drop),
                    "window_index": i,
                }
            )

    return pd.DataFrame(breakpoints)


def build_orthogonal_factors(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    factor_definitions: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build orthogonal factors from feature groups.

    Args:
        features: DataFrame with feature columns
        labels: Series with target labels
        factor_definitions: Optional mapping of factor names to feature lists.
            If None, uses default factor definitions based on feature blocks.

    Returns:
        Tuple of (factor_values DataFrame, factor_ic DataFrame)
    """
    if features.empty or labels.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Default factor definitions based on feature blocks
    if factor_definitions is None:
        factor_definitions = {
            "risk_factor": [],  # Market risk features
            "value_factor": [],  # P/E, fundamental features
            "quality_factor": [],  # Profitability, ownership features
            "sentiment_factor": [],  # Narrative, sentiment features
            "liquidity_factor": [],  # Volume, delivery features
            "momentum_factor": [],  # Momentum, technical features
        }

        # Map features to factors based on block prefixes
        for col in features.columns:
            if "__" in col:
                block = col.split("__")[0]
                if block in ["ownership", "fundamental"]:
                    if "pe" in col.lower() or "eps" in col.lower() or "value" in col.lower():
                        factor_definitions["value_factor"].append(col)
                    else:
                        factor_definitions["quality_factor"].append(col)
                elif block in ["narrative", "sentiment"]:
                    factor_definitions["sentiment_factor"].append(col)
                elif block in ["turnover", "delivery"]:
                    factor_definitions["liquidity_factor"].append(col)
                elif block in ["momentum", "technical", "regime_adaptive"]:
                    factor_definitions["momentum_factor"].append(col)
                elif block in ["regime", "volatility"]:
                    factor_definitions["risk_factor"].append(col)

    # Build factor values (first principal component or mean of features)
    factor_values = pd.DataFrame(index=features.index)

    for factor_name, factor_features in factor_definitions.items():
        if not factor_features:
            continue

        # Get available features
        available_features = [f for f in factor_features if f in features.columns]
        if not available_features:
            continue

        # Simple approach: mean of standardized features
        factor_data = features[available_features].copy()
        # Standardize
        factor_data = (factor_data - factor_data.mean()) / (factor_data.std() + 1e-9)
        # Take mean (could use PCA for true orthogonality)
        factor_values[factor_name] = factor_data.mean(axis=1)

    # Compute IC for each factor
    factor_ics = []
    for factor_name in factor_values.columns:
        factor_vals = factor_values[factor_name].dropna()
        aligned_labels = labels.reindex(factor_vals.index).dropna()
        factor_vals = factor_vals.reindex(aligned_labels.index)

        if len(factor_vals) < 10:
            continue

        ic = compute_ic(factor_vals, aligned_labels, method="pearson")
        rank_ic = compute_ic(factor_vals, aligned_labels, method="spearman")

        factor_ics.append(
            {
                "factor": factor_name,
                "feature_count": len([f for f in factor_definitions.get(factor_name, []) if f in features.columns]),
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )

    factor_ic_df = pd.DataFrame(factor_ics)

    # Compute cross-factor correlations to check orthogonality
    if len(factor_values.columns) > 1:
        factor_corr = factor_values.corr()
        factor_ic_df["max_correlation"] = [
            float(factor_corr.loc[factor, factor_corr.columns != factor].abs().max())
            if factor in factor_corr.index
            else np.nan
            for factor in factor_ic_df["factor"]
        ]

    return factor_values, factor_ic_df


def analyze_cross_feature_correlation(
    features: pd.DataFrame,
    *,
    correlation_threshold: float = 0.8,
    min_cluster_size: int = 2,
) -> pd.DataFrame:
    """Analyze cross-feature correlations and identify redundant feature groups.

    Args:
        features: DataFrame with feature columns
        correlation_threshold: Minimum correlation to be considered redundant
        min_cluster_size: Minimum features per cluster

    Returns:
        DataFrame with correlation analysis results
    """
    if features.empty:
        return pd.DataFrame()

    feature_cols = [col for col in features.columns if col not in ["date", "ticker"]]
    if len(feature_cols) < 2:
        return pd.DataFrame()

    # Compute correlation matrix
    corr_matrix = features[feature_cols].corr().abs()

    # Find clusters of highly correlated features
    clusters = {}
    used_features = set()

    for feat in feature_cols:
        if feat in used_features:
            continue

        # Find all features highly correlated with this one
        correlated = corr_matrix[feat][corr_matrix[feat] > correlation_threshold].index.tolist()
        cluster = [f for f in correlated if f not in used_features]

        if len(cluster) >= min_cluster_size:
            clusters[feat] = cluster
            used_features.update(cluster)

    # Build results
    results = []
    for representative, cluster_members in clusters.items():
        # Compute average correlation within cluster
        cluster_corr = corr_matrix.loc[cluster_members, cluster_members]
        avg_corr = float(cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].mean())

        results.append(
            {
                "representative_feature": representative,
                "cluster_size": len(cluster_members),
                "cluster_members": ", ".join(cluster_members),
                "avg_correlation": avg_corr,
                "max_correlation": float(cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].max()),
            }
        )

    return pd.DataFrame(results).sort_values("cluster_size", ascending=False)


def analyze_ic_by_regime(
    features: pd.DataFrame,
    labels: pd.Series,
    prices: pd.DataFrame,
    cfg: ICDiagnosticsConfig = None,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
) -> pd.DataFrame:
    """Comprehensive IC analysis decomposed by market regime.

    This is an enhanced version that provides detailed IC statistics for each
    feature across different regime combinations (bull/bear × low/mid/high vol).

    Args:
        features: DataFrame with feature columns
        labels: Series with target labels
        prices: DataFrame with price panel (date, ticker, close)
        cfg: IC diagnostics configuration
        date_col: Column name for date in prices
        ticker_col: Column name for ticker in prices
        close_col: Column name for close price in prices

    Returns:
        DataFrame with IC statistics by feature and regime
    """
    if cfg is None:
        cfg = ICDiagnosticsConfig()

    if features.empty or labels.empty or prices.empty:
        return pd.DataFrame()

    # Compute regime flags (reuse logic from audit module)
    from intentflow_ai.features.audit import compute_regime_flags_for_audit, FeatureAuditConfig

    audit_cfg = FeatureAuditConfig(
        regime_vol_buckets=cfg.regime_vol_buckets,
        risk_cfg=cfg.risk_cfg,
        min_samples_per_regime=cfg.min_samples_per_regime,
    )

    regime_flags = compute_regime_flags_for_audit(
        prices, audit_cfg, ticker_col=ticker_col, close_col=close_col, date_col=date_col
    )

    # Merge features with regime flags
    # This is simplified - in practice you'd need proper alignment
    features_with_regime = features.copy()
    
    # Try to extract date from features
    if "date" in features.columns:
        features_with_regime["date"] = pd.to_datetime(features["date"])
    else:
        # Assume features index aligns with some date structure
        # For now, use first available date (this should be improved)
        logger.warning("Date column not found in features, attempting alignment")
        if "date" in prices.columns:
            # Try to merge with prices to get dates
            # This is a simplified approach - real implementation would need proper alignment
            features_with_regime["date"] = pd.to_datetime(regime_flags["date"].iloc[0])
        else:
            return pd.DataFrame()

    # Merge regime flags
    features_with_regime = features_with_regime.merge(
        regime_flags[["date", "regime_is_bull", "regime_is_bear", "vol_regime", "index_vol_pct"]],
        on="date",
        how="left",
    )

    # Align labels
    aligned_labels = labels.reindex(features.index).dropna()

    results = []

    # Define regime combinations
    regime_combinations = [
        ("bull", "low_vol"),
        ("bull", "mid_vol"),
        ("bull", "high_vol"),
        ("bear", "low_vol"),
        ("bear", "mid_vol"),
        ("bear", "high_vol"),
    ]

    feature_cols = [col for col in features.columns if col not in ["date", "ticker"]]

    # Compute IC for each feature in each regime
    for trend, vol in regime_combinations:
        trend_col = f"regime_is_{trend}"
        regime_mask = (
            (features_with_regime[trend_col] == 1) & (features_with_regime["vol_regime"] == vol)
        )

        if regime_mask.sum() < cfg.min_samples_per_regime:
            continue

        regime_features = features_with_regime.loc[regime_mask]
        regime_labels = aligned_labels.reindex(regime_features.index).dropna()
        regime_features = regime_features.reindex(regime_labels.index)

        if regime_features.empty or regime_labels.empty:
            continue

        # Compute IC for each feature in this regime
        for feat_col in feature_cols:
            if feat_col not in regime_features.columns:
                continue

            feat_values = regime_features[feat_col].dropna()
            aligned_labels_subset = regime_labels.reindex(feat_values.index).dropna()
            feat_values = feat_values.reindex(aligned_labels_subset.index)

            if len(feat_values) < cfg.min_samples_per_regime:
                continue

            # Return IC (Pearson - signal magnitude)
            return_ic = compute_return_ic(feat_values, aligned_labels_subset)
            # Rank IC (Spearman - rank correlation)
            rank_ic = compute_ic(feat_values, aligned_labels_subset, method="spearman")

            if pd.isna(return_ic):
                continue

            results.append(
                {
                    "regime_trend": trend,
                    "regime_vol": vol,
                    "regime_combined": f"{trend}_{vol}",
                    "feature": feat_col,
                    "sample_count": len(feat_values),
                    "return_ic": return_ic,
                    "ic": return_ic,  # Legacy alias
                    "rank_ic": rank_ic,
                    "ic_abs": abs(return_ic),
                }
            )

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Add summary statistics
    summary = df_results.groupby("regime_combined")["return_ic"].agg(["mean", "median", "std", "count"]).reset_index()
    summary.columns = ["regime_combined", "regime_mean_ic", "regime_median_ic", "regime_std_ic", "regime_feature_count"]

    df_results = df_results.merge(summary, on="regime_combined", how="left")

    return df_results

