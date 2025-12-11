"""
Modern Market Structure Features

Features designed to capture post-2020 market dynamics:
- Passive investing dominance
- Index concentration
- Momentum crowding
- Monthly SIP/MF flow patterns

These features help the model adapt to changing market structure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def compute_index_concentration(
    prices: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    volume_col: str = "volume",
    top_n: int = 10,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Compute index concentration features.
    
    Measures how concentrated the market is in top stocks by volume/price.
    High concentration = more passive flow vulnerability.
    
    Features:
    - top10_volume_share: % of total volume in top 10 stocks
    - top10_volume_share_chg: Change in concentration over lookback
    - hhi_volume: Herfindahl index of volume distribution
    """
    df = prices.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Daily market-wide metrics
    daily_vol = df.groupby([date_col, ticker_col])[volume_col].sum().reset_index()
    
    results = []
    
    for date in df[date_col].unique():
        day_data = daily_vol[daily_vol[date_col] == date]
        
        if len(day_data) < 20:
            continue
            
        total_vol = day_data[volume_col].sum()
        if total_vol == 0:
            continue
        
        # Sort by volume
        sorted_data = day_data.sort_values(volume_col, ascending=False)
        
        # Top N concentration
        top_n_vol = sorted_data.head(top_n)[volume_col].sum()
        top_n_share = top_n_vol / total_vol
        
        # HHI (concentration index)
        shares = day_data[volume_col] / total_vol
        hhi = (shares ** 2).sum()
        
        results.append({
            date_col: date,
            "top10_volume_share": top_n_share,
            "hhi_volume": hhi,
        })
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) > lookback:
        # Rolling change in concentration
        result_df["top10_volume_share_chg"] = (
            result_df["top10_volume_share"] - 
            result_df["top10_volume_share"].shift(lookback)
        )
    else:
        result_df["top10_volume_share_chg"] = 0.0
    
    return result_df


def compute_breadth_divergence(
    prices: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    windows: list = [5, 20, 60],
) -> pd.DataFrame:
    """
    Compute breadth divergence features.
    
    Measures divergence between cap-weighted and equal-weighted returns.
    Large divergence = narrow market leadership, vulnerable to rotation.
    
    Features:
    - breadth_divergence_{window}: EW return - proxy cap-weighted return
    - advance_decline_ratio: % of stocks up vs down
    - pct_above_50dma: % of stocks above their 50-day MA
    """
    df = prices.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Pivot to get price matrix
    price_matrix = df.pivot_table(
        index=date_col, columns=ticker_col, values=close_col
    )
    
    # Compute returns
    returns = price_matrix.pct_change()
    
    results = pd.DataFrame(index=price_matrix.index)
    
    # Equal-weight daily return
    ew_ret = returns.mean(axis=1)
    
    # Proxy cap-weight: volume-weighted return (approximate)
    # For simplicity, we use median return as contrast
    median_ret = returns.median(axis=1)
    
    for window in windows:
        # Rolling returns
        ew_cumret = (1 + ew_ret).rolling(window).apply(lambda x: x.prod() - 1, raw=True)
        med_cumret = (1 + median_ret).rolling(window).apply(lambda x: x.prod() - 1, raw=True)
        
        # Divergence
        results[f"breadth_divergence_{window}d"] = ew_cumret - med_cumret
    
    # Advance/Decline ratio
    results["advance_decline_ratio"] = (returns > 0).sum(axis=1) / returns.notna().sum(axis=1)
    
    # % above 50 DMA
    ma_50 = price_matrix.rolling(50).mean()
    results["pct_above_50dma"] = (price_matrix > ma_50).sum(axis=1) / price_matrix.notna().sum(axis=1)
    
    # % above 200 DMA
    ma_200 = price_matrix.rolling(200).mean()
    results["pct_above_200dma"] = (price_matrix > ma_200).sum(axis=1) / price_matrix.notna().sum(axis=1)
    
    return results.reset_index()


def compute_momentum_crowding(
    prices: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    momentum_window: int = 60,
    top_n: int = 20,
    corr_window: int = 20,
) -> pd.DataFrame:
    """
    Compute momentum crowding features.
    
    Measures correlation among top momentum stocks.
    High crowding = risk of simultaneous reversal.
    
    Features:
    - momentum_crowding: Avg correlation of top momentum stocks
    - momentum_dispersion: Std of momentum across stocks
    - momentum_reversal_risk: Crowding × momentum intensity
    """
    df = prices.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Pivot to get price matrix
    price_matrix = df.pivot_table(
        index=date_col, columns=ticker_col, values=close_col
    )
    
    # Compute momentum (60-day returns)
    returns = price_matrix.pct_change()
    momentum = price_matrix.pct_change(momentum_window)
    
    results = []
    dates = price_matrix.index[momentum_window + corr_window:]
    
    for date in dates:
        try:
            # Get momentum ranking at this date
            mom_today = momentum.loc[date].dropna()
            if len(mom_today) < top_n * 2:
                continue
            
            # Top momentum stocks
            top_mom_tickers = mom_today.nlargest(top_n).index.tolist()
            
            # Get recent returns for correlation
            start_date = price_matrix.index[price_matrix.index <= date][-corr_window]
            recent_returns = returns.loc[start_date:date, top_mom_tickers]
            
            if recent_returns.shape[1] < 5 or recent_returns.shape[0] < 10:
                continue
            
            # Pairwise correlation
            corr_matrix = recent_returns.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            avg_corr = corr_matrix.where(mask).stack().mean()
            
            # Momentum dispersion
            mom_dispersion = mom_today.std()
            
            # Momentum intensity (average of top momentum)
            mom_intensity = mom_today.nlargest(top_n).mean()
            
            results.append({
                date_col: date,
                "momentum_crowding": avg_corr,
                "momentum_dispersion": mom_dispersion,
                "momentum_reversal_risk": avg_corr * abs(mom_intensity),
            })
        except Exception:
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def compute_passive_flow_patterns(
    prices: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute passive flow pattern features.
    
    SIP and mutual fund flows are concentrated at month-end/start.
    Large caps benefit more from passive flows.
    
    Features:
    - days_to_month_end: Days until month end
    - is_month_end_week: Binary flag for last week of month
    - is_month_first_week: Binary flag for first week of month
    - is_expiry_week: Binary flag for F&O expiry week (last Thursday)
    """
    df = prices[[date_col]].drop_duplicates().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Days to month end
    df["month_end"] = df[date_col] + pd.offsets.MonthEnd(0)
    df["days_to_month_end"] = (df["month_end"] - df[date_col]).dt.days
    
    # Normalize to 0-1
    df["days_to_month_end_norm"] = df["days_to_month_end"] / 31.0
    
    # Month end/start week flags
    df["is_month_end_week"] = (df["days_to_month_end"] <= 5).astype(float)
    df["is_month_start"] = (df[date_col].dt.day <= 5).astype(float)
    
    # Quarter end flag (stronger flows)
    df["is_quarter_end"] = (
        df[date_col].dt.month.isin([3, 6, 9, 12]) & 
        (df["days_to_month_end"] <= 3)
    ).astype(float)
    
    # F&O expiry week (last Thursday of month)
    # Approximate: if within 7 days of month end and day >= 22
    df["is_expiry_week"] = (
        (df[date_col].dt.day >= 22) & (df["days_to_month_end"] <= 7)
    ).astype(float)
    
    return df[[date_col, "days_to_month_end_norm", "is_month_end_week", 
               "is_month_start", "is_quarter_end", "is_expiry_week"]]


def compute_volatility_regime_features(
    prices: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    windows: list = [20, 60, 252],
) -> pd.DataFrame:
    """
    Compute volatility regime features.
    
    Features:
    - market_vol_{window}d: Market-wide realized volatility
    - vol_percentile_{window}d: Current vol percentile in rolling window
    - vol_regime: Low/Mid/High classification
    - vol_trend: Rising/Falling/Stable
    """
    df = prices.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Market-wide returns (equal-weight)
    daily_ret = df.groupby(date_col)[close_col].mean().pct_change()
    
    results = pd.DataFrame(index=daily_ret.index)
    
    for window in windows:
        # Realized volatility (annualized)
        vol = daily_ret.rolling(window).std() * np.sqrt(252)
        results[f"market_vol_{window}d"] = vol
        
        # Percentile rank
        results[f"vol_percentile_{window}d"] = vol.rolling(252).rank(pct=True)
    
    # Vol regime classification
    vol_60 = results["market_vol_60d"]
    vol_33 = vol_60.quantile(0.33)
    vol_66 = vol_60.quantile(0.66)
    
    results["vol_regime_score"] = np.where(
        vol_60 < vol_33, 0,  # Low vol
        np.where(vol_60 < vol_66, 1, 2)  # Mid/High vol
    )
    
    # Vol trend (20d change)
    results["vol_trend"] = results["market_vol_20d"] / results["market_vol_20d"].shift(20) - 1
    
    return results.reset_index()


def build_modern_features(
    prices: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Build all modern market features for the given price panel.
    
    Returns a DataFrame indexed by date with all modern features.
    """
    logger.info("Building modern market features...")
    
    all_features = []
    
    # 1. Index concentration (if volume available)
    if volume_col in prices.columns:
        try:
            conc = compute_index_concentration(prices, date_col, ticker_col, close_col, volume_col)
            all_features.append(conc)
            logger.info(f"  Added index concentration: {len(conc.columns)-1} features")
        except Exception as e:
            logger.warning(f"  Index concentration failed: {e}")
    
    # 2. Breadth divergence
    try:
        breadth = compute_breadth_divergence(prices, date_col, ticker_col, close_col)
        all_features.append(breadth)
        logger.info(f"  Added breadth divergence: {len(breadth.columns)-1} features")
    except Exception as e:
        logger.warning(f"  Breadth divergence failed: {e}")
    
    # 3. Momentum crowding
    try:
        crowding = compute_momentum_crowding(prices, date_col, ticker_col, close_col)
        if len(crowding) > 0:
            all_features.append(crowding)
            logger.info(f"  Added momentum crowding: {len(crowding.columns)-1} features")
    except Exception as e:
        logger.warning(f"  Momentum crowding failed: {e}")
    
    # 4. Passive flow patterns
    try:
        passive = compute_passive_flow_patterns(prices, date_col)
        all_features.append(passive)
        logger.info(f"  Added passive flow patterns: {len(passive.columns)-1} features")
    except Exception as e:
        logger.warning(f"  Passive flow patterns failed: {e}")
    
    # 5. Volatility regime
    try:
        vol_regime = compute_volatility_regime_features(prices, date_col, ticker_col, close_col)
        all_features.append(vol_regime)
        logger.info(f"  Added volatility regime: {len(vol_regime.columns)-1} features")
    except Exception as e:
        logger.warning(f"  Volatility regime failed: {e}")
    
    # Merge all features on date
    if not all_features:
        logger.warning("No modern features could be computed")
        return pd.DataFrame()
    
    result = all_features[0]
    for feat_df in all_features[1:]:
        result = result.merge(feat_df, on=date_col, how="outer")
    
    logger.info(f"Total modern features: {len(result.columns)-1}")
    
    return result


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    
    # Load prices
    prices = pd.read_parquet(ROOT / "data" / "processed" / "prices.parquet")
    print(f"Loaded {len(prices)} rows of price data")
    
    # Build features
    modern = build_modern_features(prices)
    print(f"\nBuilt modern features: {modern.shape}")
    print(modern.head(10))
