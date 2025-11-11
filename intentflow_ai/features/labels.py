"""Label engineering helpers for swing predictions."""

from __future__ import annotations

import pandas as pd


def make_excess_label(
    df: pd.DataFrame,
    horizon_days: int = 10,
    sector_col: str = "sector",
    thresh: float = 0.015,
) -> pd.DataFrame:
    """Compute excess forward returns vs sector and derive binary labels."""

    required = {"ticker", "close", "date", sector_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for label generation: {', '.join(sorted(missing))}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    fwd_col = f"fwd_ret_{horizon_days}"
    future_price = data.groupby("ticker")["close"].shift(-horizon_days)
    data[fwd_col] = (future_price - data["close"]) / data["close"]

    sector_key = [ "date", sector_col ]
    data["sector_fwd"] = data.groupby(sector_key)[fwd_col].transform("mean")
    data["excess_fwd"] = data[fwd_col] - data["sector_fwd"]
    data["label"] = (data["excess_fwd"] >= thresh).astype(int)
    return data
