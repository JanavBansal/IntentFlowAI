"""Universe helpers for ticker mappings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def load_universe(path: Path) -> pd.DataFrame:
    """Load the configured universe CSV and validate basic schema."""

    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    df = pd.read_csv(path)
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
