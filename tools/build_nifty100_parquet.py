
"""Build a NIFTY100 OHLCV parquet using yfinance.

This is a temporary helper to make the NIFTY100 universe the working sandbox
while the long-term target remains a full NIFTY200 PIT panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from intentflow_ai.data.universe import _normalize_ticker  # reuse normalization
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NIFTY100 daily OHLCV via yfinance and write parquet.")
    parser.add_argument(
        "--universe",
        default="data/external/universe/nifty100_universe.csv",
        help="CSV with `ticker` column of NSE symbols (no suffix).",
    )
    parser.add_argument(
        "--output",
        default="data/raw/price_confirmation/data.parquet",
        help="Parquet path to write the combined panel.",
    )
    parser.add_argument(
        "--start",
        default="2017-01-01",
        help="Start date for history (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional end date.",
    )
    return parser.parse_args()


def load_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Universe CSV must include a `ticker` column.")
    tickers = df["ticker"].dropna().astype(str).map(_normalize_ticker).unique().tolist()
    tickers = [t for t in tickers if t]
    if not tickers:
        raise ValueError("Universe CSV contained no tickers.")
    return tickers


def fetch_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    symbol = f"{ticker}.NS"
    # Don't use group_by for single ticker - it causes MultiIndex issues
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, actions=False)
    if df.empty:
        raise ValueError("Empty dataframe returned.")
    if isinstance(df.columns, pd.MultiIndex):
        # Handle MultiIndex columns
        if len(df.columns.get_level_values(1).unique()) == 1:
            # Single ticker, just use first level
            df.columns = df.columns.get_level_values(0)
        else:
            # Multiple tickers (shouldn't happen for single symbol, but handle it)
            if symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1, drop_level=True)
            elif ticker in df.columns.get_level_values(1):
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            else:
                # Take first ticker's data
                first_ticker = df.columns.get_level_values(1)[0]
                df = df.xs(first_ticker, axis=1, level=1, drop_level=True)
    # Reset index to get date as column
    df = df.reset_index()
    # Rename columns to lowercase
    df = df.rename(columns=str.lower)
    # Map column names to expected format - prefer Close over Adj Close
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})
    elif "adj close" in df.columns and "close" in df.columns:
        # Keep Close, drop Adj Close
        df = df.drop(columns=["adj close"])
    # Ensure date column exists (it should be from reset_index)
    if "date" not in df.columns:
        # If index name is Date, it should have been converted
        if df.index.name and "date" in str(df.index.name).lower():
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
        else:
            raise ValueError("Date column not found after reset_index")
    expected_cols = {"date", "open", "high", "low", "close", "volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}")
    out = df[list(expected_cols)].copy()
    out["ticker"] = ticker
    out = out[["date", "ticker", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["close"]).drop_duplicates(["date", "ticker"])
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values(["date", "ticker"])


def main() -> None:
    args = parse_args()
    universe_path = Path(args.universe)
    output_path = Path(args.output)

    tickers = load_universe(universe_path)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for t in tickers:
        try:
            df = fetch_ticker(t, start=args.start, end=args.end)
            if df.empty:
                skipped.append(t)
                logger.warning("Skipping empty fetch", extra={"ticker": t})
                continue
            frames.append(df)
            logger.info("Fetched ticker", extra={"ticker": t, "rows": len(df)})
        except Exception as exc:  # pragma: no cover - network path
            skipped.append(t)
            logger.warning("Failed to fetch ticker", extra={"ticker": t, "error": str(exc)})

    if not frames:
        raise SystemExit("No data fetched; check universe and network access.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    dates = pd.to_datetime(combined["date"])
    print(
        f"Written {len(combined):,} rows for {combined['ticker'].nunique()} tickers "
        f"from {dates.min().date()} to {dates.max().date()} -> {output_path}"
    )
    if skipped:
        print(f"Skipped {len(skipped)} tickers: {', '.join(sorted(set(skipped)))}")


if __name__ == "__main__":
    main()
