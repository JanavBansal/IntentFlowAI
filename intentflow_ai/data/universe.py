"""Universe helpers for ticker mappings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def _shim_universe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept simplified CSVs and expand required columns."""

    lower_cols = {c.lower(): c for c in df.columns}
    required = {"ticker_nse", "ticker_yf"}
    if required.issubset(lower_cols.keys()):
        return df

    ticker_col = lower_cols.get("ticker")
    if ticker_col:
        ticker_series = df[ticker_col].astype(str).str.upper().str.strip()
        df["ticker_nse"] = ticker_series
        df["ticker_yf"] = ticker_series

    sector_col = lower_cols.get("sector")
    if sector_col and "sector" not in df.columns:
        df["sector"] = df[sector_col]
    return df


def load_universe(path: Path | str) -> pd.DataFrame:
    """Load the configured universe CSV and validate basic schema."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    df = pd.read_csv(path)
    df = _shim_universe_columns(df)
    # Allow simplified schema by injecting required columns.
    if {"ticker", "sector"}.issubset(df.columns) and not {"ticker_nse", "ticker_yf"}.issubset(df.columns):
        df = df.copy()
        ticker_norm = df["ticker"].astype(str).str.strip().str.upper()
        df["ticker_nse"] = ticker_norm
        df["ticker_yf"] = ticker_norm
    validate_universe(df)
    logger.info("Loaded universe", extra={"tickers": len(df)})
    return df


def validate_universe(df: pd.DataFrame) -> None:
    """Ensure the universe file is well-formed."""

    required = ["ticker_nse", "ticker_yf", "sector"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Universe missing columns: {missing_cols}")

    normalized = df[required].copy()
    normalized = normalized.apply(lambda col: col.astype(str).str.strip())
    if normalized.isnull().any().any():
        raise ValueError("Universe contains null values in required columns.")

    if normalized["ticker_yf"].duplicated().any():
        dups = normalized.loc[normalized["ticker_yf"].duplicated(), "ticker_yf"].tolist()
        raise ValueError(f"Universe has duplicate ticker_yf entries: {dups}")

    sector_counts = normalized["sector"].value_counts().to_dict()
    logger.info("Universe sector distribution", extra={"counts": sector_counts})


def load_universe_membership(path: Path) -> pd.DataFrame:
    """Load historical membership windows (start/end dates per ticker)."""

    if not path.exists():
        raise FileNotFoundError(f"Universe membership file not found: {path}")
    df = pd.read_csv(path)
    required = ["ticker_nse", "start_date", "end_date"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Membership file missing columns: {missing}")
    df["ticker_nse"] = df["ticker_nse"].astype(str).str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    if df["start_date"].isna().any():
        raise ValueError("Membership start_date contains nulls.")
    logger.info("Loaded membership history", extra={"rows": len(df)})
    return df


def apply_membership_filter(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Filter price rows to periods when tickers are part of the configured universe."""

    if membership.empty:
        return prices

    membership = membership.copy()
    membership["end_date"] = membership["end_date"].fillna(pd.Timestamp.max)
    membership = membership.rename(columns={"ticker_nse": "ticker"})
    merged = prices.merge(
        membership[["ticker", "start_date", "end_date"]],
        on="ticker",
        how="left",
    )
    mask = merged["start_date"].isna() | (
        (merged["date"] >= merged["start_date"]) & (merged["date"] <= merged["end_date"])
    )
    filtered = merged.loc[mask, prices.columns]
    filtered = filtered.drop_duplicates(subset=["date", "ticker"])
    return filtered
