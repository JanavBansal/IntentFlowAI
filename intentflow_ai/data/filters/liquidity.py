"""
Liquidity Filter for Universe Selection

Filters out illiquid stocks that cannot be traded at modeled prices.
Critical for realistic backtesting and live trading.
"""

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LiquidityConfig:
    """Configuration for liquidity filtering."""
    
    # Minimum average daily volume (shares)
    min_avg_volume: int = 100_000
    
    # Minimum average daily turnover (INR)
    # Default: 50 Lakh = 5,000,000
    min_avg_turnover: float = 5_000_000
    
    # Lookback period for calculating averages
    lookback_days: int = 20
    
    # Minimum trading days in lookback (to avoid newly listed stocks)
    min_trading_days: int = 15
    
    # Maximum bid-ask spread proxy (daily range / close)
    max_spread_proxy: float = 0.05  # 5%
    
    # Minimum market cap (INR) - optional
    min_market_cap: Optional[float] = None


class LiquidityFilter:
    """
    Filter illiquid stocks from the universe.
    
    Applies multiple liquidity criteria:
    1. Minimum average daily volume
    2. Minimum average daily turnover (volume * price)
    3. Minimum trading days
    4. Maximum spread proxy (high-low range)
    
    Usage:
        filter = LiquidityFilter(LiquidityConfig(min_avg_turnover=10_000_000))
        filtered_df = filter.filter(price_df)
        
    Or for universe selection:
        liquid_tickers = filter.get_liquid_tickers(price_df, as_of_date)
    """
    
    def __init__(self, config: Optional[LiquidityConfig] = None):
        self.config = config or LiquidityConfig()
    
    def filter(self, df: pd.DataFrame, as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Filter DataFrame to only include liquid stocks.
        
        Args:
            df: Price DataFrame with columns [date, ticker, close, volume, high, low]
            as_of_date: If provided, filter based on liquidity as of this date
            
        Returns:
            Filtered DataFrame with only liquid stocks
        """
        if df.empty:
            return df
        
        liquid_tickers = self.get_liquid_tickers(df, as_of_date)
        
        if not liquid_tickers:
            logger.warning("No liquid tickers found! Check liquidity thresholds.")
            return df
        
        filtered = df[df["ticker"].isin(liquid_tickers)].copy()
        
        logger.info(
            "Liquidity filter applied",
            extra={
                "original_tickers": df["ticker"].nunique(),
                "liquid_tickers": len(liquid_tickers),
                "filtered_out": df["ticker"].nunique() - len(liquid_tickers),
            }
        )
        
        return filtered
    
    def get_liquid_tickers(
        self, 
        df: pd.DataFrame, 
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Set[str]:
        """
        Get set of tickers that pass liquidity filters.
        
        Args:
            df: Price DataFrame
            as_of_date: Date to evaluate liquidity (uses latest if None)
            
        Returns:
            Set of liquid ticker symbols
        """
        if df.empty:
            return set()
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # Determine evaluation date
        if as_of_date is None:
            as_of_date = df["date"].max()
        else:
            as_of_date = pd.to_datetime(as_of_date)
        
        # Get lookback window
        lookback_start = as_of_date - pd.Timedelta(days=self.config.lookback_days * 1.5)
        window_df = df[(df["date"] >= lookback_start) & (df["date"] <= as_of_date)]
        
        if window_df.empty:
            logger.warning(f"No data in lookback window ending {as_of_date}")
            return set()
        
        # Calculate liquidity metrics per ticker
        liquidity_metrics = self._calculate_liquidity_metrics(window_df)
        
        # Apply filters
        liquid_tickers = self._apply_liquidity_filters(liquidity_metrics)
        
        return liquid_tickers
    
    def _calculate_liquidity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity metrics for each ticker."""
        
        metrics = []
        
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date")
            
            # Number of trading days
            trading_days = len(group)
            
            # Average daily volume
            avg_volume = group["volume"].mean() if "volume" in group.columns else 0
            
            # Average daily turnover (volume * close)
            if "volume" in group.columns and "close" in group.columns:
                group["turnover"] = group["volume"] * group["close"]
                avg_turnover = group["turnover"].mean()
            else:
                avg_turnover = 0
            
            # Spread proxy: average (high - low) / close
            if all(col in group.columns for col in ["high", "low", "close"]):
                group["spread_proxy"] = (group["high"] - group["low"]) / group["close"].replace(0, np.nan)
                avg_spread = group["spread_proxy"].mean()
            else:
                avg_spread = 0
            
            # Volume stability (coefficient of variation)
            if "volume" in group.columns and avg_volume > 0:
                volume_cv = group["volume"].std() / avg_volume
            else:
                volume_cv = np.nan
            
            metrics.append({
                "ticker": ticker,
                "trading_days": trading_days,
                "avg_volume": avg_volume,
                "avg_turnover": avg_turnover,
                "avg_spread": avg_spread,
                "volume_cv": volume_cv,
            })
        
        return pd.DataFrame(metrics)
    
    def _apply_liquidity_filters(self, metrics: pd.DataFrame) -> Set[str]:
        """Apply all liquidity filters and return passing tickers."""
        
        if metrics.empty:
            return set()
        
        # Start with all tickers
        passing = metrics.copy()
        initial_count = len(passing)
        
        # Filter 1: Minimum trading days
        passing = passing[passing["trading_days"] >= self.config.min_trading_days]
        after_days = len(passing)
        
        # Filter 2: Minimum average volume
        passing = passing[passing["avg_volume"] >= self.config.min_avg_volume]
        after_volume = len(passing)
        
        # Filter 3: Minimum average turnover
        passing = passing[passing["avg_turnover"] >= self.config.min_avg_turnover]
        after_turnover = len(passing)
        
        # Filter 4: Maximum spread proxy
        passing = passing[passing["avg_spread"] <= self.config.max_spread_proxy]
        after_spread = len(passing)
        
        logger.debug(
            "Liquidity filter breakdown",
            extra={
                "initial": initial_count,
                "after_min_days": after_days,
                "after_min_volume": after_volume,
                "after_min_turnover": after_turnover,
                "after_max_spread": after_spread,
            }
        )
        
        return set(passing["ticker"].tolist())
    
    def estimate_market_impact(
        self, 
        ticker: str, 
        order_value: float, 
        adv: float
    ) -> float:
        """
        Estimate market impact (slippage) for a given order.
        
        Uses square-root market impact model:
        Impact = base_impact * sqrt(participation_rate)
        
        Args:
            ticker: Stock ticker
            order_value: Order value in INR
            adv: Average daily value traded
            
        Returns:
            Estimated impact as a fraction (e.g., 0.005 = 0.5%)
        """
        if adv <= 0:
            return 0.02  # 2% default for illiquid stocks
        
        participation_rate = order_value / adv
        
        # Square-root impact model
        # Base impact ~10bps, scales with sqrt of participation
        base_impact = 0.001  # 0.1%
        impact = base_impact * np.sqrt(1 + participation_rate * 100)
        
        # Cap at 2%
        return min(impact, 0.02)
    
    def get_liquidity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a composite liquidity score for each ticker.
        
        Score ranges from 0 (illiquid) to 100 (highly liquid).
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with ticker and liquidity_score columns
        """
        metrics = self._calculate_liquidity_metrics(df)
        
        if metrics.empty:
            return pd.DataFrame(columns=["ticker", "liquidity_score"])
        
        # Normalize each metric to 0-100 scale
        metrics["volume_score"] = self._percentile_score(metrics["avg_volume"])
        metrics["turnover_score"] = self._percentile_score(metrics["avg_turnover"])
        metrics["spread_score"] = 100 - self._percentile_score(metrics["avg_spread"])  # Lower is better
        metrics["stability_score"] = 100 - self._percentile_score(metrics["volume_cv"])  # Lower CV is better
        
        # Composite score (weighted average)
        metrics["liquidity_score"] = (
            metrics["turnover_score"] * 0.4 +
            metrics["volume_score"] * 0.3 +
            metrics["spread_score"] * 0.2 +
            metrics["stability_score"] * 0.1
        )
        
        return metrics[["ticker", "liquidity_score", "avg_turnover", "avg_volume"]]
    
    @staticmethod
    def _percentile_score(series: pd.Series) -> pd.Series:
        """Convert series to percentile scores (0-100)."""
        return series.rank(pct=True) * 100


def create_liquidity_filter(
    min_turnover_lakhs: float = 50,
    min_volume: int = 100_000,
    lookback_days: int = 20
) -> LiquidityFilter:
    """
    Convenience function to create a liquidity filter.
    
    Args:
        min_turnover_lakhs: Minimum average daily turnover in lakhs (1 lakh = 100,000)
        min_volume: Minimum average daily volume in shares
        lookback_days: Days to look back for calculating averages
        
    Returns:
        Configured LiquidityFilter instance
    """
    config = LiquidityConfig(
        min_avg_turnover=min_turnover_lakhs * 100_000,
        min_avg_volume=min_volume,
        lookback_days=lookback_days,
    )
    return LiquidityFilter(config)
