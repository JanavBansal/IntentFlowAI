"""Coverage reporting for price history."""

from __future__ import annotations

import pandas as pd


def price_coverage_report(df: pd.DataFrame, *, min_days: int | None = None) -> dict:
    """Compute per-ticker coverage stats and summary."""

    if df.empty or "ticker" not in df.columns:
        per_ticker = pd.DataFrame(
            columns=["ticker", "start_date", "end_date", "num_days", "missing_close_pct"]
        )
        summary = {"num_tickers": 0, "median_days": 0, "symbols_below_min": 0}
        return {"per_ticker": per_ticker, "summary": summary}

    grouped = df.groupby("ticker", dropna=True)
    records = []
    for ticker, g in grouped:
        g = g.sort_values("date")
        start = g["date"].min()
        end = g["date"].max()
        num_days = g["date"].nunique()
        missing = float(g["close"].isna().mean())
        records.append(
            {
                "ticker": ticker,
                "start_date": start,
                "end_date": end,
                "num_days": num_days,
                "missing_close_pct": missing,
            }
        )
    per_ticker = pd.DataFrame(records)
    threshold = min_days if min_days is not None else per_ticker["num_days"].median()
    summary = {
        "num_tickers": int(per_ticker["ticker"].nunique()),
        "median_days": float(per_ticker["num_days"].median()) if not per_ticker.empty else 0,
        "symbols_below_min": int((per_ticker["num_days"] < threshold).sum()),
    }
    return {"per_ticker": per_ticker, "summary": summary}
