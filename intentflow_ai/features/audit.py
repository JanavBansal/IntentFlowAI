"""Feature audit tools for regime-based analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from intentflow_ai.backtest.filters import RiskFilterConfig, compute_regime_flags
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureAuditConfig:
    """Configuration for feature audit."""

    regime_vol_buckets: List[float] = None  # e.g., [0, 33, 67, 100] for low/mid/high vol
    risk_cfg: RiskFilterConfig = None
    min_samples_per_regime: int = 100  # Minimum samples needed to compute stats

    def __post_init__(self):
        if self.regime_vol_buckets is None:
            self.regime_vol_buckets = [0, 33, 67, 100]  # Low, Mid, High
        if self.risk_cfg is None:
            self.risk_cfg = RiskFilterConfig()


def compute_regime_flags_for_audit(
    prices: pd.DataFrame,
    cfg: FeatureAuditConfig,
    *,
    ticker_col: str = "ticker",
    close_col: str = "close",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute regime flags for feature audit."""
    if "date" not in prices.columns:
        raise ValueError(f"prices DataFrame must have 'date' column")
    if ticker_col not in prices.columns:
        raise ValueError(f"prices DataFrame must have '{ticker_col}' column")
    if close_col not in prices.columns:
        raise ValueError(f"prices DataFrame must have '{close_col}' column")

    px_pivot = prices.pivot_table(index=date_col, columns=ticker_col, values=close_col)
    if px_pivot.empty:
        logger.warning("Empty price pivot, cannot compute regime flags")
        return pd.DataFrame()

    px_pivot.index = pd.to_datetime(px_pivot.index)
    regime_flags = compute_regime_flags(px_pivot, cfg.risk_cfg)

    # Reset index to make date a column
    regime_flags = regime_flags.reset_index()
    if "index" in regime_flags.columns:
        regime_flags.rename(columns={"index": "date"}, inplace=True)
    elif regime_flags.index.name:
        regime_flags.rename_axis("date", inplace=True)
        regime_flags = regime_flags.reset_index()

    if "date" not in regime_flags.columns:
        regime_flags["date"] = pd.to_datetime(regime_flags.index)
    else:
        regime_flags["date"] = pd.to_datetime(regime_flags["date"])

    # Determine bull/bear from trend_ok
    regime_flags["regime_is_bull"] = regime_flags["trend_ok"].astype(int)
    regime_flags["regime_is_bear"] = (~regime_flags["trend_ok"]).astype(int)

    # Create volatility buckets from index_vol
    vol_buckets = cfg.regime_vol_buckets
    if len(vol_buckets) < 2:
        vol_buckets = [0, 33, 67, 100]

    # Compute volatility percentile for each date (expanding window)
    vol_series = regime_flags["index_vol"]
    vol_pct = vol_series.expanding(min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= 20 else np.nan,
        raw=False,
    )
    regime_flags["index_vol_pct"] = vol_pct.values

    # Classify volatility regime
    def classify_vol_pct(pct):
        if pd.isna(pct):
            return "unknown"
        for i in range(len(vol_buckets) - 1):
            if vol_buckets[i] <= pct < vol_buckets[i + 1]:
                if i == 0:
                    return "low_vol"
                elif i == 1:
                    return "mid_vol"
                else:
                    return "high_vol"
        return "high_vol"

    regime_flags["vol_regime"] = regime_flags["index_vol_pct"].apply(classify_vol_pct)

    return regime_flags


def audit_features_by_regime(
    features: pd.DataFrame,
    labels: pd.Series,
    prices: pd.DataFrame,
    cfg: FeatureAuditConfig = None,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
) -> pd.DataFrame:
    """Audit feature importance and behavior across different market regimes.

    Args:
        features: DataFrame with feature columns
        labels: Series with target labels (aligned with features index)
        prices: DataFrame with price panel (date, ticker, close)
        cfg: Feature audit configuration
        date_col: Column name for date in prices
        ticker_col: Column name for ticker in prices
        close_col: Column name for close price in prices

    Returns:
        DataFrame with feature audit results by regime
    """
    if cfg is None:
        cfg = FeatureAuditConfig()

    if features.empty or labels.empty or prices.empty:
        return pd.DataFrame()

    # Compute regime flags
    regime_flags = compute_regime_flags_for_audit(
        prices, cfg, ticker_col=ticker_col, close_col=close_col, date_col=date_col
    )

    # Merge features with regime flags
    # Assume features index aligns with a date/ticker structure
    # We need to merge on date
    features_with_regime = features.copy()
    
    # Try to extract date from features index or columns
    if "date" in features.columns:
        features_with_regime["date"] = pd.to_datetime(features["date"])
    elif hasattr(features.index, "get_level_values"):
        # MultiIndex with date level
        try:
            features_with_regime["date"] = pd.to_datetime(features.index.get_level_values("date"))
        except (KeyError, AttributeError):
            logger.warning("Could not extract date from features index, using first available date")
            features_with_regime["date"] = regime_flags["date"].iloc[0]
    else:
        # Try to merge with prices to get dates
        if "ticker" in features.columns and date_col in prices.columns:
            # Merge features with prices to get dates
            prices_subset = prices[[date_col, ticker_col]].drop_duplicates()
            # This is a simplified merge - in practice, you'd need proper alignment
            features_with_regime["date"] = pd.to_datetime(regime_flags["date"].iloc[0])
        else:
            logger.warning("Could not determine date alignment, using first regime date")
            features_with_regime["date"] = pd.to_datetime(regime_flags["date"].iloc[0])

    # Merge regime flags
    features_with_regime = features_with_regime.merge(
        regime_flags[["date", "regime_is_bull", "regime_is_bear", "vol_regime", "index_vol_pct"]],
        on="date",
        how="left",
    )

    # Align labels with features
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

    # Compute feature statistics for each regime
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

        # Compute feature importance (IC) for each feature
        feature_cols = [col for col in features.columns if col not in ["date", "ticker"]]
        
        for feat_col in feature_cols:
            if feat_col not in regime_features.columns:
                continue
                
            feat_values = regime_features[feat_col].dropna()
            aligned_labels_subset = regime_labels.reindex(feat_values.index).dropna()
            feat_values = feat_values.reindex(aligned_labels_subset.index)

            if len(feat_values) < cfg.min_samples_per_regime:
                continue

            # Information Coefficient (Pearson correlation)
            try:
                ic = float(feat_values.corr(aligned_labels_subset, method="pearson"))
            except Exception:
                ic = np.nan

            # Rank IC (Spearman correlation)
            try:
                rank_ic = float(feat_values.corr(aligned_labels_subset, method="spearman"))
            except Exception:
                rank_ic = np.nan

            # Feature statistics
            feat_mean = float(feat_values.mean())
            feat_std = float(feat_values.std())
            feat_median = float(feat_values.median())

            results.append(
                {
                    "regime_trend": trend,
                    "regime_vol": vol,
                    "regime_combined": f"{trend}_{vol}",
                    "feature": feat_col,
                    "sample_count": len(feat_values),
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "feat_mean": feat_mean,
                    "feat_std": feat_std,
                    "feat_median": feat_median,
                }
            )

    return pd.DataFrame(results)


def audit_feature_stability(
    features: pd.DataFrame,
    labels: pd.Series,
    dates: pd.Series,
    *,
    window_days: int = 252,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Compute rolling IC over time to assess feature stability.

    Args:
        features: DataFrame with feature columns
        labels: Series with target labels
        dates: Series with dates (aligned with features index)
        window_days: Rolling window size in days
        min_samples: Minimum samples per window

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
    results = []

    # Rolling window analysis
    unique_dates = aligned["date"].unique()
    window_size = pd.Timedelta(days=window_days)

    for i in range(len(unique_dates)):
        window_end = unique_dates[i]
        window_start = window_end - window_size

        window_data = aligned[(aligned["date"] >= window_start) & (aligned["date"] <= window_end)]

        if len(window_data) < min_samples:
            continue

        window_labels = window_data["label"]

        # Compute IC for each feature in this window
        feature_cols = [col for col in features.columns if col not in ["date", "ticker"]]
        for feat_col in feature_cols:
            if feat_col not in window_data.columns:
                continue

            feat_values = window_data[feat_col].dropna()
            aligned_labels = window_labels.reindex(feat_values.index).dropna()
            feat_values = feat_values.reindex(aligned_labels.index)

            if len(feat_values) < min_samples:
                continue

            try:
                ic = float(feat_values.corr(aligned_labels, method="pearson"))
                rank_ic = float(feat_values.corr(aligned_labels, method="spearman"))
            except Exception:
                ic = np.nan
                rank_ic = np.nan

            results.append(
                {
                    "window_end_date": window_end,
                    "feature": feat_col,
                    "sample_count": len(feat_values),
                    "ic": ic,
                    "rank_ic": rank_ic,
                }
            )

    return pd.DataFrame(results)


def identify_feature_clusters(
    features: pd.DataFrame,
    *,
    correlation_threshold: float = 0.8,
    min_cluster_size: int = 2,
) -> Dict[str, List[str]]:
    """Identify clusters of highly correlated features.

    Args:
        features: DataFrame with feature columns
        correlation_threshold: Minimum correlation to be in same cluster
        min_cluster_size: Minimum features per cluster

    Returns:
        Dictionary mapping cluster representative to list of features in cluster
    """
    if features.empty:
        return {}

    # Compute correlation matrix
    feature_cols = [col for col in features.columns if col not in ["date", "ticker"]]
    corr_matrix = features[feature_cols].corr().abs()

    # Find clusters using simple greedy approach
    clusters = {}
    used_features = set()

    for feat in feature_cols:
        if feat in used_features:
            continue

        # Find all features highly correlated with this one
        cluster = [feat]
        correlated = corr_matrix[feat][corr_matrix[feat] > correlation_threshold].index.tolist()
        cluster.extend([f for f in correlated if f != feat and f not in used_features])

        if len(cluster) >= min_cluster_size:
            clusters[feat] = cluster
            used_features.update(cluster)

    return clusters

