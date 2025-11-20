"""Universe helpers for ticker mappings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)

# Centralized override map for known NSE/Yahoo symbol renames.
DEFAULT_SYMBOL_OVERRIDES: dict[str, str] = {
    "ADANITRANS": "ADANIENSOL",  # Adani Transmission → Adani Energy Solutions
    "MOTHERSUMI": "MOTHERSON",  # Old Motherson Sumi ticker
}


def _normalize_ticker(value: str | int | float) -> str:
    return str(value).strip().upper()


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


def load_universe(
    path: Path | str,
    *,
    as_of: str | datetime | None = None,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    membership_path: Path | str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load the configured universe CSV, apply overrides, and return PIT subset.

    Args:
        path: Base universe CSV path (must include ticker/sector columns).
        as_of: Optional single date to fetch the active universe for.
        start_date: Optional inclusive start when requesting a range.
        end_date: Optional inclusive end when requesting a range.
        membership_path: Optional CSV containing membership history
            (ticker_nse,start_date,end_date[,sector]).
        overrides: Optional mapping of ticker overrides (e.g. ADANITRANS→ADANIENSOL).

    Returns:
        DataFrame with ticker_nse/ticker_yf/sector and start_date/end_date columns,
        filtered to the requested window when provided.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    df = pd.read_csv(path)
    df = _shim_universe_columns(df)
    # Allow simplified schema by injecting required columns.
    if {"ticker", "sector"}.issubset(df.columns) and not {"ticker_nse", "ticker_yf"}.issubset(df.columns):
        df = df.copy()
        ticker_norm = df["ticker"].apply(_normalize_ticker)
        df["ticker_nse"] = ticker_norm
        df["ticker_yf"] = ticker_norm

    override_map = {**DEFAULT_SYMBOL_OVERRIDES, **(overrides or {})}
    df["ticker_nse"] = df["ticker_nse"].apply(_normalize_ticker).map(lambda t: override_map.get(t, t))
    df["ticker_yf"] = df["ticker_yf"].apply(_normalize_ticker).map(lambda t: override_map.get(t, t))
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip()
    validate_universe(df)

    membership: Optional[pd.DataFrame] = None
    if membership_path:
        try:
            membership = load_universe_membership(Path(membership_path), overrides=override_map)
        except FileNotFoundError:
            logger.warning("Membership file missing; falling back to static universe.", extra={"path": membership_path})
    result = _point_in_time_universe(
        base=df,
        membership=membership,
        as_of=as_of,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info(
        "Loaded universe",
        extra={
            "tickers": result["ticker_nse"].nunique(),
            "as_of": str(as_of) if as_of else None,
            "start": str(start_date) if start_date else None,
            "end": str(end_date) if end_date else None,
        },
    )
    return result


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


def load_universe_membership(path: Path, *, overrides: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Load historical membership windows (start/end dates per ticker)."""

    if not path.exists():
        raise FileNotFoundError(f"Universe membership file not found: {path}")
    df = pd.read_csv(path)
    required = ["ticker_nse", "start_date", "end_date"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Membership file missing columns: {missing}")
    override_map = {**DEFAULT_SYMBOL_OVERRIDES, **(overrides or {})}
    df["ticker_nse"] = df["ticker_nse"].apply(_normalize_ticker).map(lambda t: override_map.get(t, t))
    if "ticker_yf" in df.columns:
        df["ticker_yf"] = df["ticker_yf"].apply(_normalize_ticker).map(lambda t: override_map.get(t, t))
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.tz_localize(None)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.tz_localize(None)
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


def _point_in_time_universe(
    base: pd.DataFrame,
    membership: Optional[pd.DataFrame],
    *,
    as_of: str | datetime | None = None,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
) -> pd.DataFrame:
    """Combine static base universe with membership windows for PIT slicing."""

    base = base.copy()
    base["ticker"] = base["ticker_nse"]
    base["start_date"] = pd.to_datetime(start_date).tz_localize(None) if start_date else pd.Timestamp("1900-01-01")
    base["end_date"] = pd.to_datetime(end_date).tz_localize(None) if end_date else pd.Timestamp("2100-01-01")

    if membership is not None and not membership.empty:
        membership = membership.copy()
        membership["start_date"] = membership["start_date"].fillna(pd.Timestamp("1900-01-01"))
        membership["end_date"] = membership["end_date"].fillna(pd.Timestamp("2100-01-01"))
        membership["ticker"] = membership["ticker_nse"]
        merged = membership.merge(
            base[["ticker_nse", "ticker_yf", "sector"]].drop_duplicates("ticker_nse"),
            on="ticker_nse",
            how="left",
        )
    else:
        merged = base

    for col in ["start_date", "end_date"]:
        merged[col] = pd.to_datetime(merged[col], errors="coerce").dt.tz_localize(None)

    if as_of:
        as_of_ts = pd.to_datetime(as_of).tz_localize(None)
        mask = (merged["start_date"] <= as_of_ts) & (merged["end_date"] >= as_of_ts)
        merged = merged.loc[mask]
    else:
        if start_date:
            start_ts = pd.to_datetime(start_date).tz_localize(None)
            merged = merged.loc[merged["end_date"] >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date).tz_localize(None)
            merged = merged.loc[merged["start_date"] <= end_ts]

    merged = merged.drop_duplicates(subset=["ticker_nse", "start_date", "end_date"])
    return merged.reset_index(drop=True)
