"""
Fundamental Features for Earnings Oracle Agent.

Uses EODHD fundamental data to compute features for quality screening.
This module handles point-in-time correct merging of fundamentals with price data.

Features:
- Valuation: PE, PB, PS, EV/EBITDA, sector-relative PE
- Profitability: ROE, ROA, gross margin, operating margin, net margin
- Quality: Accruals ratio, cash flow to net income
- Growth: Revenue growth, earnings growth
- Leverage: Debt to equity, current ratio
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def load_eodhd_fundamentals(cache_path: str = "data/cache/fundamentals/eodhd_full.parquet") -> pd.DataFrame:
    """
    Load cached fundamental data — EODHD preferred, screener.in fallback.

    Falls back to screener_consolidated.parquet when the EODHD cache is missing
    or stale (>90 days).  Screener.in provides PE, ROE, D/E, current_ratio,
    growth rates — enough for the Earnings Oracle's ±0.15 signal cap.
    """
    from pathlib import Path
    from datetime import datetime

    SCREENER_PATH = Path("data/cache/fundamentals/screener_consolidated.parquet")

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        for col in ["date", "available_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    # Try EODHD first (if file exists and is < 90 days old)
    eodhd_path = Path(cache_path)
    if eodhd_path.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(eodhd_path.stat().st_mtime)).days
        if age_days < 90:
            return _load(eodhd_path)
        logger.info(f"EODHD cache is {age_days} days old — trying screener.in fallback")

    # Screener.in fallback
    if SCREENER_PATH.exists():
        logger.info(f"Loading fundamentals from screener.in: {SCREENER_PATH}")
        df = _load(SCREENER_PATH)
        # Screener.in uses 'symbol'; normalise to match EODHD schema
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        # Add available_date if missing (use date + 45-day reporting delay)
        if "available_date" not in df.columns and "date" in df.columns:
            df["available_date"] = df["date"] + pd.Timedelta(days=45)
        return df

    logger.warning(
        f"No fundamental data found at {cache_path} or {SCREENER_PATH}. "
        "Run scripts/refresh_screener_fundamentals.py to restore fundamentals."
    )
    return pd.DataFrame()


def merge_fundamentals_pit(
    price_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    ticker_col: str = 'ticker',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Merge fundamentals with price data using point-in-time correct logic.
    
    Uses available_date (not report_date) to ensure no look-ahead bias.
    For each (ticker, date) in price_df, gets the most recent fundamental
    data that was available by that date.
    
    Args:
        price_df: Price/feature DataFrame with ticker and date columns
        fund_df: Fundamental DataFrame with symbol, date, and available_date
        ticker_col: Name of ticker column in price_df
        date_col: Name of date column in price_df
        
    Returns:
        price_df with fundamental features appended
    """
    if fund_df.empty:
        logger.warning("Empty fundamentals DataFrame. Adding zero fundamental features.")
        return _add_zero_fundamental_features(price_df)
    
    # Prepare fundamentals for merge
    fund = fund_df.copy()
    fund = fund.rename(columns={'symbol': ticker_col})
    
    # Use available_date for point-in-time correctness
    if 'available_date' in fund.columns:
        fund['merge_date'] = fund['available_date']
    else:
        fund['merge_date'] = fund['date']
    
    # Select columns to merge
    fundamental_cols = get_fundamental_feature_names()
    available_cols = [c for c in fundamental_cols if c in fund.columns]
    
    if not available_cols:
        logger.warning("No fundamental columns found in data.")
        return _add_zero_fundamental_features(price_df)
    
    merge_cols = [ticker_col, 'merge_date'] + available_cols
    fund = fund[merge_cols].drop_duplicates(subset=[ticker_col, 'merge_date'])
    
    # Sort by ticker first, then date (required for merge_asof by group)
    price_df = price_df.sort_values([ticker_col, date_col]).reset_index(drop=True)
    fund = fund.sort_values([ticker_col, 'merge_date']).reset_index(drop=True)
    
    # Ensure date columns are datetime
    price_df[date_col] = pd.to_datetime(price_df[date_col])
    fund['merge_date'] = pd.to_datetime(fund['merge_date'])
    
    # Use grouped merge_asof for better handling
    results = []
    for ticker in price_df[ticker_col].unique():
        price_ticker = price_df[price_df[ticker_col] == ticker].copy()
        fund_ticker = fund[fund[ticker_col] == ticker].copy()
        
        if fund_ticker.empty:
            # No fundamentals for this ticker
            results.append(price_ticker)
            continue
        
        # merge_asof for this ticker
        merged = pd.merge_asof(
            price_ticker.sort_values(date_col),
            fund_ticker.sort_values('merge_date'),
            left_on=date_col,
            right_on='merge_date',
            direction='backward',
            suffixes=('', '_fund')
        )
        results.append(merged)
    
    if results:
        result = pd.concat(results, ignore_index=True)
    else:
        result = price_df.copy()
    
    # Drop merge_date column
    if 'merge_date' in result.columns:
        result = result.drop(columns=['merge_date'])
    
    # Fill missing fundamental features with NaN
    for col in fundamental_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    logger.info(f"Merged fundamentals: {len(available_cols)} features, {result[available_cols[0]].notna().sum()} non-null values")
    
    return result


def compute_sector_relative_features(
    df: pd.DataFrame,
    ticker_col: str = 'ticker',
    sector_col: str = 'sector'
) -> pd.DataFrame:
    """
    Compute sector-relative fundamental features.
    
    Adds features like:
    - pe_sector_relative: PE / sector median PE
    - roe_sector_percentile: ROE percentile within sector
    
    These help identify relative value within sectors.
    """
    result = df.copy()
    
    if sector_col not in result.columns:
        logger.debug("No sector column found. Skipping sector-relative features.")
        return result
    
    features_to_normalize = ['pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity']
    
    for feat in features_to_normalize:
        if feat not in result.columns:
            continue
        
        # Sector median
        sector_median = result.groupby(sector_col)[feat].transform('median')
        
        # Sector relative (ratio to median)
        rel_col = f'{feat}_sector_rel'
        result[rel_col] = result[feat] / sector_median.replace(0, np.nan)
        result[rel_col] = result[rel_col].replace([np.inf, -np.inf], np.nan)
        
        # Sector percentile
        pct_col = f'{feat}_sector_pct'
        result[pct_col] = result.groupby(sector_col)[feat].rank(pct=True)
    
    return result


def compute_growth_features(df: pd.DataFrame, ticker_col: str = 'ticker') -> pd.DataFrame:
    """
    Compute growth features from fundamentals.
    
    Adds:
    - revenue_growth_qoq: Quarter-over-quarter revenue growth
    - net_income_growth_qoq: Quarter-over-quarter earnings growth
    - revenue_growth_yoy: Year-over-year revenue growth (if enough data)
    """
    result = df.copy()
    
    if 'revenue' not in result.columns:
        return result
    
    # Sort by ticker and date for proper growth calculation
    result = result.sort_values([ticker_col, 'date'])
    
    # QoQ growth (1 quarter lag = roughly 90 days)
    for col in ['revenue', 'net_income', 'operating_income']:
        if col not in result.columns:
            continue
        
        growth_col = f'{col}_growth_qoq'
        result[growth_col] = result.groupby(ticker_col)[col].pct_change()
        result[growth_col] = result[growth_col].clip(-1, 5)  # Cap extreme values
    
    # YoY growth (4 quarters)
    for col in ['revenue', 'net_income']:
        if col not in result.columns:
            continue
        
        yoy_col = f'{col}_growth_yoy'
        result[yoy_col] = result.groupby(ticker_col)[col].pct_change(4)
        result[yoy_col] = result[yoy_col].clip(-1, 10)
    
    return result


def get_fundamental_feature_names() -> List[str]:
    """Return list of all fundamental feature names expected for Earnings Oracle."""
    return [
        # Valuation
        'pe_ratio',
        'pb_ratio',
        'ps_ratio',
        'peg_ratio',
        'ev_ebitda',
        'ev_revenue',
        'dividend_yield',
        
        # Profitability
        'roe',
        'roa',
        'gross_margin',
        'operating_margin',
        'net_margin',
        
        # Quality
        'accruals_ratio',
        'cf_to_net_income',
        
        # Leverage
        'debt_to_equity',
        'debt_to_assets',
        'current_ratio',
        'quick_ratio',
        
        # ESG
        'esg_total',
        'esg_governance',
        
        # Sector-relative (computed)
        'pe_ratio_sector_rel',
        'roe_sector_pct',
    ]


def _add_zero_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add placeholder fundamental features with NaN values."""
    result = df.copy()
    for col in get_fundamental_feature_names():
        if col not in result.columns:
            result[col] = np.nan
    return result


def prepare_fundamentals_for_training(
    price_df: pd.DataFrame,
    fund_cache_path: str = "data/cache/fundamentals/eodhd_full.parquet",
    add_sector_relative: bool = True,
    add_growth: bool = True
) -> pd.DataFrame:
    """
    Main entry point: Prepare and merge fundamentals for model training.
    
    This is point-in-time safe and can be used in walk-forward optimization.
    
    Args:
        price_df: Price/feature DataFrame with 'ticker' and 'date' columns
        fund_cache_path: Path to EODHD parquet cache
        add_sector_relative: Whether to add sector-relative features
        add_growth: Whether to add growth features
        
    Returns:
        DataFrame with fundamental features merged
    """
    # Load fundamentals
    fund_df = load_eodhd_fundamentals(fund_cache_path)
    
    if fund_df.empty:
        logger.warning("No fundamentals loaded. Proceeding without fundamental features.")
        return _add_zero_fundamental_features(price_df)
    
    # Merge point-in-time
    result = merge_fundamentals_pit(price_df, fund_df)
    
    # Add derived features
    if add_sector_relative and 'sector' in result.columns:
        result = compute_sector_relative_features(result)
    
    if add_growth:
        result = compute_growth_features(result)
    
    logger.info(f"Prepared fundamentals: {len(result)} rows with {len(get_fundamental_feature_names())} fundamental features")
    
    return result
