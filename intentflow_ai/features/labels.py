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
    data["sector_fwd"] = sector_group.transform("mean")

    data["excess_fwd"] = data[fwd_col] - data["sector_fwd"]
    data["label"] = (data["excess_fwd"] >= thresh).astype(int)
    return data
