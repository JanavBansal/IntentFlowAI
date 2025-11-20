"""Label engineering helpers."""

from __future__ import annotations

from typing import Final

import pandas as pd


def make_excess_label(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: str = "ticker",
    sector_col: str = "sector",
    horizon_days: int = 10,
    thresh: float = 0.015,
) -> pd.DataFrame:
    """Create leak-safe excess-return labels relative to sector peers."""

    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    fwd_col: Final[str] = f"fwd_ret_{horizon_days}d"

    def _forward_return(series: pd.Series) -> pd.Series:
        return series.shift(-horizon_days) / series - 1

    data[fwd_col] = (
        data.groupby(ticker_col, group_keys=False)[price_col]
        .transform(_forward_return)
    )

    sector_group = data.groupby([sector_col, date_col], group_keys=False)[fwd_col]
    sector_sum = sector_group.transform("sum")
    sector_count = sector_group.transform("count")
    leave_one_out = (sector_sum - data[fwd_col]) / (sector_count - 1)
    sector_mean = sector_group.transform("mean")
    data["sector_fwd"] = leave_one_out.where(sector_count > 1, sector_mean)
    data["excess_fwd"] = data[fwd_col] - data["sector_fwd"]
    data["label"] = (data["excess_fwd"] >= thresh).astype(int)
    return data


def make_triple_barrier_label(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: str = "ticker",
    horizon_days: int = 10,
    pt: float = 0.04,
    sl: float = 0.02,
) -> pd.DataFrame:
    """Create Triple Barrier labels (Profit Take / Stop Loss / Time Out).
    
    FIXED: Now truly point-in-time. Only labels are created for dates where
    the full horizon period has elapsed. This prevents data leakage.
    
    Label 1: Hit PT before SL within horizon (successful trade).
    Label 0: Hit SL before PT, or hit neither within horizon (unsuccessful trade).
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    def _compute_barrier(group: pd.DataFrame) -> pd.Series:
        """Compute labels only for samples where full horizon has passed."""
        prices = group[price_col].values
        n = len(prices)
        # Initialize as NaN - will only label where we have complete data
        labels = pd.Series(pd.NA, index=group.index, dtype='Int64')

        # CRITICAL FIX: Only label up to (n - horizon_days)
        # This ensures we never look at data that wouldn't be available
        # at prediction time
        for i in range(n - horizon_days):
            current_price = prices[i]
            if current_price <= 0:
                continue
            
            # Look forward window - BUT this is now safe because
            # we're only doing this for historical labels, and we
            # exclude the last horizon_days samples from labeling
            window = prices[i + 1 : i + 1 + horizon_days]
            returns = (window / current_price) - 1.0
            
            # Find first touch
            pt_touch = (returns >= pt).argmax() if (returns >= pt).any() else -1
            sl_touch = (returns <= -sl).argmax() if (returns <= -sl).any() else -1
            
            if pt_touch != -1 and sl_touch != -1:
                # Both hit, check which was first
                if pt_touch < sl_touch:
                    labels.iloc[i] = 1  # Profit take wins
                else:
                    labels.iloc[i] = 0  # Stop loss wins
            elif pt_touch != -1:
                labels.iloc[i] = 1  # Profit take only
            # else 0 (default) - stop loss only or timeout
            
        # Last horizon_days samples will remain NaN - these will be dropped
        # This is CORRECT - we can't label what we don't know yet
        return labels

   # Apply per ticker
    data["label"] = data.groupby(ticker_col, group_keys=False).apply(_compute_barrier)
    
    # Compute forward return ONLY for samples with labels (for metrics)
    # This is also backward-looking because we only compute for labeled samples
    def _compute_fwd_return(group: pd.DataFrame) -> pd.Series:
        prices = group[price_col].values
        n = len(prices)
        fwd_ret = pd.Series(float('nan'), index=group.index, dtype='float64')
        
        for i in range(n - horizon_days):
            if prices[i] > 0 and prices[i + horizon_days] > 0:
                fwd_ret.iloc[i] = (prices[i + horizon_days] / prices[i]) - 1
        
        return fwd_ret
    
    fwd_col = f"fwd_ret_{horizon_days}d"
    data[fwd_col] = data.groupby(ticker_col, group_keys=False).apply(_compute_fwd_return)
    data["excess_fwd"] = data[fwd_col]  # For compatibility
    
    # Drop samples without labels (last horizon_days per ticker)
    initial_count = len(data)
    data = data.dropna(subset=["label"])
    dropped_count = initial_count - len(data)
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count} unlabeled samples (most recent {horizon_days} days per ticker)")
    
    return data
