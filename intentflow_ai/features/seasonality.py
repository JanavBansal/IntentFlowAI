"""
Seasonality Features

Indian market-specific seasonal patterns:
- Budget season (January-February)
- Earnings seasons (quarterly)
- Diwali rally (October-November)
- March tax-loss selling
- Monthly F&O expiry effects

These patterns have predictive power for Indian equities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SeasonalityConfig:
    """Configuration for seasonality features."""
    
    # Budget season (Union Budget typically presented late January/early February)
    budget_start_month: int = 1
    budget_end_month: int = 2
    
    # Earnings seasons (Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar)
    # Results typically announced 30-45 days after quarter end
    earnings_announcement_months: List[int] = None  # [1, 4, 7, 10]
    
    # Diwali period (typically late October to mid-November)
    diwali_start_month: int = 10
    diwali_end_month: int = 11
    
    # Tax-loss harvesting period
    tax_selling_month: int = 3
    
    def __post_init__(self):
        if self.earnings_announcement_months is None:
            self.earnings_announcement_months = [1, 4, 7, 10]


# Approximate Diwali dates (can be updated annually)
DIWALI_DATES = {
    2015: date(2015, 11, 11),
    2016: date(2016, 10, 30),
    2017: date(2017, 10, 19),
    2018: date(2018, 11, 7),
    2019: date(2019, 10, 27),
    2020: date(2020, 11, 14),
    2021: date(2021, 11, 4),
    2022: date(2022, 10, 24),
    2023: date(2023, 11, 12),
    2024: date(2024, 11, 1),
    2025: date(2025, 10, 21),
    2026: date(2026, 11, 8),
}


def get_seasonality_features(
    dt: datetime | str | pd.Timestamp,
    config: Optional[SeasonalityConfig] = None,
) -> Dict[str, Any]:
    """
    Compute seasonality features for a given date.
    
    Args:
        dt: Date to compute features for
        config: Optional configuration
        
    Returns:
        Dictionary of seasonality features
    """
    if config is None:
        config = SeasonalityConfig()
    
    dt = pd.to_datetime(dt)
    
    features = {}
    
    # Basic calendar features
    features["month"] = dt.month
    features["quarter"] = dt.quarter
    features["day_of_week"] = dt.dayofweek  # 0=Monday, 6=Sunday
    features["day_of_month"] = dt.day
    features["week_of_year"] = dt.isocalendar()[1]
    
    # Binary seasonality flags
    features["is_january"] = 1.0 if dt.month == 1 else 0.0
    features["is_december"] = 1.0 if dt.month == 12 else 0.0
    features["is_monday"] = 1.0 if dt.dayofweek == 0 else 0.0
    features["is_friday"] = 1.0 if dt.dayofweek == 4 else 0.0
    
    # Budget season
    features["is_budget_season"] = 1.0 if dt.month in [config.budget_start_month, config.budget_end_month] else 0.0
    
    # Pre-budget rally (December-January)
    features["is_pre_budget"] = 1.0 if dt.month == 12 or (dt.month == 1 and dt.day < 25) else 0.0
    
    # Earnings season
    features["is_earnings_season"] = 1.0 if dt.month in config.earnings_announcement_months else 0.0
    
    # Diwali period
    features["is_diwali_period"] = 1.0 if is_diwali_window(dt) else 0.0
    
    # Pre-Diwali rally (Muhurat trading period typically sees buying)
    features["is_pre_diwali"] = 1.0 if is_pre_diwali(dt) else 0.0
    
    # March tax selling
    features["is_march_selloff"] = 1.0 if dt.month == 3 and dt.day > 15 else 0.0
    
    # Year-end rally (November-December)
    features["is_year_end_rally"] = 1.0 if dt.month in [11, 12] else 0.0
    
    # Days to F&O monthly expiry (last Thursday of month)
    features["days_to_expiry"] = days_to_monthly_expiry(dt)
    features["is_expiry_week"] = 1.0 if features["days_to_expiry"] <= 5 else 0.0
    
    # Quarter-end effects
    features["is_quarter_end"] = 1.0 if (dt.month % 3 == 0) and dt.day > 20 else 0.0
    
    # Monsoon season (June-September) - affects certain sectors
    features["is_monsoon"] = 1.0 if dt.month in [6, 7, 8, 9] else 0.0
    
    # Wedding season (November-February) - affects consumer sectors
    features["is_wedding_season"] = 1.0 if dt.month in [11, 12, 1, 2] else 0.0
    
    return features


def is_diwali_window(dt: pd.Timestamp, window_days: int = 14) -> bool:
    """Check if date is within Diwali window."""
    year = dt.year
    diwali_date = DIWALI_DATES.get(year)
    
    if diwali_date is None:
        # Approximate: Diwali is usually in Oct-Nov
        return dt.month in [10, 11]
    
    diwali_ts = pd.Timestamp(diwali_date)
    delta = abs((dt - diwali_ts).days)
    
    return delta <= window_days


def is_pre_diwali(dt: pd.Timestamp, days_before: int = 21) -> bool:
    """Check if date is in pre-Diwali period."""
    year = dt.year
    diwali_date = DIWALI_DATES.get(year)
    
    if diwali_date is None:
        return dt.month == 10 and dt.day < 20
    
    diwali_ts = pd.Timestamp(diwali_date)
    days_to_diwali = (diwali_ts - dt).days
    
    return 0 < days_to_diwali <= days_before


def days_to_monthly_expiry(dt: pd.Timestamp) -> int:
    """
    Calculate days to monthly F&O expiry.
    
    Monthly expiry is the last Thursday of the month.
    """
    year = dt.year
    month = dt.month
    
    # Find last Thursday of month
    # Start from last day of month
    if month == 12:
        next_month = pd.Timestamp(year + 1, 1, 1)
    else:
        next_month = pd.Timestamp(year, month + 1, 1)
    
    last_day = next_month - pd.Timedelta(days=1)
    
    # Go back to Thursday (weekday 3)
    days_since_thursday = (last_day.weekday() - 3) % 7
    last_thursday = last_day - pd.Timedelta(days=days_since_thursday)
    
    # If we've passed this month's expiry, calculate for next month
    if dt > last_thursday:
        if month == 11:
            # Special case for Nov expiry: next expiry is Jan (month 13 -> 1)
            next_month = pd.Timestamp(year + 1, 1, 1)
        elif month == 12:
            # Dec expiry: next expiry is Feb
            next_month = pd.Timestamp(year + 1, 2, 1)
        else:
            # Normal case
            next_month = pd.Timestamp(year, month + 2, 1)
        
        last_day = next_month - pd.Timedelta(days=1)
        days_since_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - pd.Timedelta(days=days_since_thursday)
    
    return max(0, (last_thursday - dt).days)


def compute_seasonal_df(
    dates: pd.DatetimeIndex,
    config: Optional[SeasonalityConfig] = None,
) -> pd.DataFrame:
    """
    Compute seasonality features for multiple dates.
    
    Args:
        dates: DatetimeIndex of dates
        config: Optional configuration
        
    Returns:
        DataFrame with seasonality features
    """
    records = []
    
    for dt in dates:
        features = get_seasonality_features(dt, config)
        features["date"] = dt
        records.append(features)
    
    df = pd.DataFrame(records)
    df = df.set_index("date")
    
    return df


def get_sector_seasonality(sector: str, month: int) -> float:
    """
    Get sector-specific seasonality score.
    
    Historical patterns by sector:
    - FMCG: Strong in wedding season (Nov-Feb)
    - Auto: Strong pre-Diwali, weak in monsoon
    - Banks: Weak in Q4 (NPA recognition)
    - IT: Strong in Q3/Q4 (US client budget cycles)
    - Pharma: Generally defensive
    - Metals: Cyclical, follows global demand
    - Real Estate: Strong in festive season
    
    Returns:
        Score from -1 (historically weak) to +1 (historically strong)
    """
    # Simplified sector seasonality scores
    SECTOR_SEASONALITY = {
        "FMCG": {1: 0.5, 2: 0.3, 3: 0.0, 4: 0.0, 5: 0.0, 6: -0.2, 7: -0.3, 8: -0.2, 9: 0.0, 10: 0.3, 11: 0.5, 12: 0.5},
        "Consumer": {1: 0.5, 2: 0.3, 3: 0.0, 4: 0.0, 5: 0.0, 6: -0.2, 7: -0.3, 8: -0.2, 9: 0.0, 10: 0.3, 11: 0.5, 12: 0.5},
        "Auto": {1: 0.2, 2: 0.2, 3: -0.3, 4: -0.2, 5: 0.0, 6: -0.3, 7: -0.4, 8: -0.3, 9: 0.2, 10: 0.5, 11: 0.4, 12: 0.3},
        "Banks": {1: -0.2, 2: -0.3, 3: -0.5, 4: 0.0, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.2, 10: 0.3, 11: 0.3, 12: 0.0},
        "Financial Services": {1: -0.2, 2: -0.3, 3: -0.4, 4: 0.0, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.2, 10: 0.3, 11: 0.3, 12: 0.0},
        "IT": {1: 0.3, 2: 0.2, 3: 0.3, 4: 0.0, 5: -0.2, 6: -0.2, 7: 0.0, 8: 0.0, 9: 0.2, 10: 0.3, 11: 0.3, 12: 0.4},
        "Technology": {1: 0.3, 2: 0.2, 3: 0.3, 4: 0.0, 5: -0.2, 6: -0.2, 7: 0.0, 8: 0.0, 9: 0.2, 10: 0.3, 11: 0.3, 12: 0.4},
        "Pharma": {1: 0.1, 2: 0.1, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
        "Healthcare": {1: 0.1, 2: 0.1, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
        "Metals": {1: 0.2, 2: 0.3, 3: 0.0, 4: 0.3, 5: 0.2, 6: 0.0, 7: -0.2, 8: -0.2, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
        "Energy": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.1, 6: 0.2, 7: 0.2, 8: 0.1, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
        "Oil & Gas": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.1, 6: 0.2, 7: 0.2, 8: 0.1, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
        "Real Estate": {1: 0.3, 2: 0.2, 3: -0.2, 4: -0.3, 5: -0.3, 6: -0.2, 7: -0.2, 8: -0.1, 9: 0.2, 10: 0.4, 11: 0.5, 12: 0.3},
    }
    
    # Default neutral seasonality
    DEFAULT = {m: 0.0 for m in range(1, 13)}
    
    sector_scores = SECTOR_SEASONALITY.get(sector, DEFAULT)
    return sector_scores.get(month, 0.0)


def add_seasonality_to_df(
    df: pd.DataFrame,
    date_col: str = "date",
    sector_col: Optional[str] = "sector",
) -> pd.DataFrame:
    """
    Add seasonality features to a DataFrame.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        sector_col: Name of sector column (for sector-specific seasonality)
        
    Returns:
        DataFrame with added seasonality features
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Filter out invalid dates
    valid_mask = df[date_col].notna()
    if not valid_mask.any():
        return df
    
    # Compute features for each unique date
    unique_dates = df.loc[valid_mask, date_col].unique()
    if len(unique_dates) == 0:
        return df
    
    seasonal_df = compute_seasonal_df(pd.DatetimeIndex(unique_dates))
    seasonal_df = seasonal_df.reset_index()
    
    # Merge
    df = df.merge(seasonal_df, on=date_col, how="left")
    
    # Add sector-specific seasonality if sector column exists
    if sector_col and sector_col in df.columns:
        # Only compute for valid dates
        def safe_sector_seasonality(row):
            try:
                if pd.isna(row[date_col]):
                    return 0.0
                return get_sector_seasonality(row[sector_col], row[date_col].month)
            except (ValueError, AttributeError):
                return 0.0
        
        df["sector_seasonality"] = df.apply(safe_sector_seasonality, axis=1)
    
    return df
