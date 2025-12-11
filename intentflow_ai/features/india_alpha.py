"""
India-Specific Alpha Features Module

Implements unique signals for the Indian stock market:
1. FII/DII Flow Features - Institutional positioning
2. F&O Expiry Calendar - Monthly expiry effects  
3. Delivery Volume Conviction - Cash segment conviction
4. PCR Sentiment - Options-based sentiment

Usage:
    features = compute_india_alpha_features(prices_df, fii_df, options_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# FII/DII Flow Features
# ============================================================================

def compute_fii_flow_features(
    fii_data: pd.DataFrame,
    lookback_short: int = 5,
    lookback_long: int = 20
) -> pd.DataFrame:
    """
    Compute FII/DII flow-based features.
    
    Args:
        fii_data: DataFrame with columns [date, fii_net, dii_net]
        lookback_short: Short-term lookback (default 5 days)
        lookback_long: Long-term lookback (default 20 days)
        
    Returns:
        DataFrame with FII/DII features indexed by date
    """
    if fii_data.empty:
        logger.warning("Empty FII data provided")
        return pd.DataFrame()
    
    df = fii_data.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()
    
    features = pd.DataFrame(index=df.index)
    
    # FII cumulative flows (short and long term)
    features["fii_flow_5d"] = df["fii_net"].rolling(lookback_short).sum()
    features["fii_flow_20d"] = df["fii_net"].rolling(lookback_long).sum()
    
    # DII cumulative flows
    features["dii_flow_5d"] = df["dii_net"].rolling(lookback_short).sum()
    features["dii_flow_20d"] = df["dii_net"].rolling(lookback_long).sum()
    
    # Flow direction signals
    features["fii_direction"] = np.sign(features["fii_flow_5d"])
    features["dii_direction"] = np.sign(features["dii_flow_5d"])
    
    # DII counterweight signal (DII buying when FII selling)
    features["dii_counterweight"] = (
        (features["fii_flow_5d"] < 0) & (features["dii_flow_5d"] > 0)
    ).astype(int)
    
    # FII/DII imbalance (positive = FII bullish, negative = DII bullish)
    total_flow = features["fii_flow_5d"].abs() + features["dii_flow_5d"].abs()
    features["flow_imbalance"] = np.where(
        total_flow > 0,
        (features["fii_flow_5d"] - features["dii_flow_5d"]) / total_flow,
        0
    )
    
    # Flow momentum (acceleration)
    features["fii_momentum"] = features["fii_flow_5d"] - features["fii_flow_5d"].shift(5)
    
    # Sustained flow signal (same direction for N days)
    fii_sign = np.sign(df["fii_net"])
    features["fii_sustained_5d"] = (
        fii_sign.rolling(5).apply(lambda x: (x == x.iloc[0]).all() if len(x) == 5 else False)
    ).fillna(0).astype(int)
    
    logger.info(f"Computed {len(features.columns)} FII/DII flow features")
    return features


# ============================================================================
# F&O Expiry Calendar Features
# ============================================================================

def get_monthly_expiry_dates(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31"
) -> pd.DatetimeIndex:
    """
    Get all monthly F&O expiry dates (last Thursday of each month).
    
    Returns:
        DatetimeIndex of expiry dates
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    expiry_dates = []
    current = start.replace(day=1)
    
    while current <= end:
        # Find last Thursday of this month
        # Go to last day of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        
        last_day = next_month - timedelta(days=1)
        
        # Walk back to Thursday (weekday 3)
        days_since_thursday = (last_day.weekday() - 3) % 7
        expiry = last_day - timedelta(days=days_since_thursday)
        
        if expiry >= start and expiry <= end:
            expiry_dates.append(expiry)
        
        # Move to next month
        current = next_month
    
    return pd.DatetimeIndex(expiry_dates)


def compute_expiry_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute F&O expiry calendar features.
    
    Args:
        dates: DatetimeIndex of trading dates
        
    Returns:
        DataFrame with expiry-related features
    """
    # Get all expiry dates covering the range
    if len(dates) == 0:
        return pd.DataFrame()
    
    start_date = dates.min() - timedelta(days=60)
    end_date = dates.max() + timedelta(days=60)
    
    expiry_dates = get_monthly_expiry_dates(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    features = pd.DataFrame(index=dates)
    
    # Days to next expiry
    days_to_expiry = []
    for date in dates:
        future_expiries = expiry_dates[expiry_dates >= date]
        if len(future_expiries) > 0:
            days = (future_expiries[0] - date).days
        else:
            days = 30  # Default
        days_to_expiry.append(days)
    
    features["days_to_expiry"] = days_to_expiry
    
    # Expiry week flag (within 5 trading days)
    features["is_expiry_week"] = (features["days_to_expiry"] <= 5).astype(int)
    
    # Rollover period (3-7 days before expiry)
    features["is_rollover_period"] = (
        (features["days_to_expiry"] >= 3) & (features["days_to_expiry"] <= 7)
    ).astype(int)
    
    # First week of month (post-expiry recovery)
    features["is_first_week"] = (features["days_to_expiry"] >= 20).astype(int)
    
    # Expiry day
    features["is_expiry_day"] = (features["days_to_expiry"] == 0).astype(int)
    
    # Days since last expiry
    days_since_expiry = []
    for date in dates:
        past_expiries = expiry_dates[expiry_dates <= date]
        if len(past_expiries) > 0:
            days = (date - past_expiries[-1]).days
        else:
            days = 0
        days_since_expiry.append(days)
    
    features["days_since_expiry"] = days_since_expiry
    
    logger.info(f"Computed {len(features.columns)} expiry calendar features")
    return features


# ============================================================================
# Delivery Volume Features
# ============================================================================

def compute_delivery_features(
    prices_df: pd.DataFrame,
    delivery_threshold_high: float = 0.60,
    delivery_threshold_low: float = 0.30
) -> pd.DataFrame:
    """
    Compute delivery volume conviction features.
    
    High delivery % = strong hands (conviction)
    Low delivery % = speculation
    
    Args:
        prices_df: DataFrame with [date, ticker, volume, delivery_volume or delivery_ratio]
        delivery_threshold_high: Threshold for high conviction
        delivery_threshold_low: Threshold for speculation
        
    Returns:
        DataFrame with delivery features
    """
    if "delivery_ratio" not in prices_df.columns:
        if "delivery_volume" in prices_df.columns and "volume" in prices_df.columns:
            prices_df = prices_df.copy()
            prices_df["delivery_ratio"] = prices_df["delivery_volume"] / prices_df["volume"]
        else:
            logger.warning("No delivery data available")
            return pd.DataFrame()
    
    df = prices_df.copy()
    
    # Per-ticker delivery features
    features_list = []
    
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").copy()
        
        feat = pd.DataFrame(index=group.index)
        feat["date"] = group["date"]
        feat["ticker"] = ticker
        
        # Raw delivery ratio
        feat["delivery_ratio"] = group["delivery_ratio"]
        
        # Delivery ratio z-score (5-day rolling)
        rolling_mean = group["delivery_ratio"].rolling(20, min_periods=5).mean()
        rolling_std = group["delivery_ratio"].rolling(20, min_periods=5).std()
        feat["delivery_zscore"] = (group["delivery_ratio"] - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # High/Low conviction flags
        feat["high_conviction"] = (group["delivery_ratio"] > delivery_threshold_high).astype(int)
        feat["low_conviction"] = (group["delivery_ratio"] < delivery_threshold_low).astype(int)
        
        # Delivery trend (5-day)
        feat["delivery_trend_5d"] = group["delivery_ratio"].pct_change(5, fill_method=None)
        
        # Volume-weighted delivery
        if "volume" in group.columns:
            feat["volume_delivery_product"] = group["delivery_ratio"] * np.log1p(group["volume"])
        
        features_list.append(feat)
    
    if features_list:
        features = pd.concat(features_list, ignore_index=True)
        logger.info(f"Computed delivery features for {df['ticker'].nunique()} tickers")
        return features
    
    return pd.DataFrame()


# ============================================================================
# Combined India Alpha Features
# ============================================================================

def compute_india_alpha_features(
    prices_df: pd.DataFrame,
    fii_df: Optional[pd.DataFrame] = None,
    add_expiry: bool = True,
    add_delivery: bool = True
) -> pd.DataFrame:
    """
    Compute all India-specific alpha features.
    
    Args:
        prices_df: Price data with [date, ticker, close, volume, ...]
        fii_df: Optional FII/DII flow data
        add_expiry: Whether to add F&O expiry features
        add_delivery: Whether to add delivery volume features
        
    Returns:
        DataFrame with all India alpha features
    """
    if "date" not in prices_df.columns:
        logger.error("prices_df must have 'date' column")
        return pd.DataFrame()
    
    dates = pd.to_datetime(prices_df["date"].unique())
    
    # Start with expiry features (date-level)
    date_features = pd.DataFrame()
    if add_expiry:
        expiry_feats = compute_expiry_features(dates)
        expiry_feats = expiry_feats.reset_index().rename(columns={"index": "date"})
        date_features = expiry_feats
    
    # Add FII features (date-level)
    if fii_df is not None and not fii_df.empty:
        fii_feats = compute_fii_flow_features(fii_df)
        fii_feats = fii_feats.reset_index().rename(columns={"index": "date"})
        
        if date_features.empty:
            date_features = fii_feats
        else:
            date_features = date_features.merge(fii_feats, on="date", how="outer")
    
    # Add delivery features (ticker-level)
    ticker_features = pd.DataFrame()
    if add_delivery and "delivery_ratio" in prices_df.columns:
        ticker_features = compute_delivery_features(prices_df)
    
    # Merge date-level features to prices
    result = prices_df[["date", "ticker"]].copy()
    
    if not date_features.empty:
        result = result.merge(date_features, on="date", how="left")
    
    if not ticker_features.empty:
        # Get only the computed features (exclude date, ticker)
        merge_cols = [c for c in ticker_features.columns if c not in ["date", "ticker"] or c in ["date", "ticker"]]
        result = result.merge(
            ticker_features[merge_cols],
            on=["date", "ticker"],
            how="left"
        )
    
    logger.info(f"Computed {len(result.columns) - 2} India alpha features")
    return result


# ============================================================================
# Testing
# ============================================================================

def test_india_alpha():
    """Test India alpha feature computation."""
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2024-12-01", freq="B")
    tickers = ["RELIANCE", "TCS", "HDFC", "INFY", "ITC"]
    
    prices_data = []
    for ticker in tickers:
        for date in dates:
            prices_data.append({
                "date": date,
                "ticker": ticker,
                "close": np.random.uniform(100, 500),
                "volume": np.random.uniform(1e6, 1e8),
                "delivery_ratio": np.random.uniform(0.2, 0.8)
            })
    
    prices_df = pd.DataFrame(prices_data)
    
    # Sample FII data
    fii_df = pd.DataFrame({
        "date": dates,
        "fii_net": np.random.normal(500, 2000, len(dates)),
        "dii_net": np.random.normal(200, 1500, len(dates))
    })
    
    # Compute features
    features = compute_india_alpha_features(prices_df, fii_df)
    
    print(f"Features shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(features.head())
    
    return features


if __name__ == "__main__":
    test_india_alpha()
