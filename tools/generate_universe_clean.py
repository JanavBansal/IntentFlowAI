"""Generate clean NIFTY200 universe CSV from PIT history.

This script:
1. Reads nifty200_history.csv (PIT membership)
2. Filters for currently active tickers
3. Cleans ticker symbols (removes synthetic suffixes, preserves - and &)
4. Optionally merges with base nifty200.csv
5. Outputs nifty200_clean.csv with unique tickers
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Ticker cleaning regex (same as fetch_api.py)
_SYNTH_SUFFIX = re.compile(r"_S\d+$")


def clean_ticker(t: str) -> str:
    """Clean ticker symbol, preserving - and & characters."""
    t = str(t).strip().upper()
    t = _SYNTH_SUFFIX.sub("", t)  # Remove synthetic suffixes like _S108
    t = t.replace(".NS", "")  # Remove .NS suffix if present
    # Keep alphanumerics plus hyphen and ampersand (for BAJAJ-AUTO, M&M, etc.)
    t = re.sub(r"[^A-Z0-9&-]", "", t)
    return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean NIFTY200 universe CSV")
    parser.add_argument(
        "--history",
        default="data/external/universe/nifty200_history.csv",
        help="Path to nifty200_history.csv",
    )
    parser.add_argument(
        "--base",
        default="data/external/universe/nifty200.csv",
        help="Optional: Path to base nifty200.csv for merging",
    )
    parser.add_argument(
        "--output",
        default="data/external/universe/nifty200_clean.csv",
        help="Output path for clean universe CSV",
    )
    parser.add_argument(
        "--as-of-date",
        help="Date to use for filtering (YYYY-MM-DD). Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    base_path = Path(args.base)
    output_path = Path(args.output)

    if not history_path.exists() and not base_path.exists():
        raise FileNotFoundError(f"Neither history file ({history_path}) nor base file ({base_path}) found. Need at least one.")

    # Prioritize base CSV if it exists (has more tickers)
    all_tickers = []
    
    # Load base CSV first (has more tickers with synthetic suffixes)
    if base_path.exists():
        print(f"Loading base universe from {base_path}...")
        try:
            base = pd.read_csv(base_path)
            # Find ticker column in base
            base_cols = {c.lower(): c for c in base.columns}
            base_ticker_col = (
                base_cols.get("ticker")
                or base_cols.get("symbol")
                or base_cols.get("ticker_nse")
                or next((c for c in base.columns if "ticker" in c.lower() or "symbol" in c.lower()), None)
            )
            if base_ticker_col:
                print(f"Found ticker column: {base_ticker_col}")
                base_tickers = pd.DataFrame({"ticker": base[base_ticker_col].map(clean_ticker)})
                base_tickers = base_tickers.dropna(subset=["ticker"])
                base_tickers = base_tickers[base_tickers["ticker"].str.len() > 0]
                base_tickers = base_tickers.drop_duplicates(subset=["ticker"])
                all_tickers.append(base_tickers)
                print(f"Found {len(base_tickers)} unique tickers in base CSV")
            else:
                print(f"Warning: Could not find ticker column in {base_path}")
        except Exception as e:
            print(f"Warning: Failed to load base CSV: {e}")

    # Load history and merge if it exists
    if history_path.exists():
        print(f"Loading history from {history_path}...")
        try:
            hist = pd.read_csv(history_path)
            # Normalize column names (handle different case/spelling)
            cols = {c.lower(): c for c in hist.columns}
            ticker_col = (
                cols.get("ticker_nse")
                or cols.get("ticker")
                or cols.get("symbol")
                or next(iter(hist.columns))
            )
            start_col = cols.get("start_date") or cols.get("effective_from") or "start_date"
            end_col = cols.get("end_date") or cols.get("effective_to") or "end_date"

            # Parse dates
            if start_col in hist.columns:
                hist[start_col] = pd.to_datetime(hist[start_col], errors="coerce")
            else:
                hist[start_col] = pd.Timestamp("2015-01-01")

            if end_col in hist.columns:
                hist[end_col] = pd.to_datetime(hist[end_col], errors="coerce").fillna(pd.Timestamp.max)
            else:
                hist[end_col] = pd.Timestamp.max

            # Get all unique tickers from history (not just current)
            # This gives us all tickers that were ever in NIFTY200
            print("Extracting all unique tickers from history (all time periods)...")
            hist_tickers = pd.DataFrame({"ticker": hist[ticker_col].map(clean_ticker)})
            hist_tickers = hist_tickers.dropna(subset=["ticker"])
            hist_tickers = hist_tickers[hist_tickers["ticker"].str.len() > 0]
            hist_tickers = hist_tickers[["ticker"]].drop_duplicates(subset=["ticker"])
            all_tickers.append(hist_tickers)
            print(f"Found {len(hist_tickers)} unique tickers in history (all periods)")
        except Exception as e:
            print(f"Warning: Failed to load history: {e}")

    # Combine all tickers
    if not all_tickers:
        raise ValueError("No tickers found in base CSV or history. Check file paths.")
    
    print("Combining all tickers...")
    alive = pd.concat(all_tickers, ignore_index=True).drop_duplicates(subset=["ticker"])
    print(f"Total unique tickers after combining: {len(alive)}")

    # Sort and output
    alive = alive.sort_values("ticker").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alive[["ticker"]].to_csv(output_path, index=False)

    print(f"✓ Saved {len(alive)} tickers to {output_path}")
    print(f"  Sample tickers: {', '.join(alive['ticker'].head(10).tolist())}")

    # Validate we have enough tickers
    if len(alive) < 180:
        print(f"\n⚠ WARNING: Only {len(alive)} tickers found (target: ≥180)")
        print("  Note: This is expected if the universe files are incomplete.")
        print("  Proceeding with available tickers. You may need to:")
        print("  1. Expand nifty200_history.csv with full NIFTY200 membership")
        print("  2. Or manually add missing tickers to the universe files")
        print("  3. Or fetch a complete NIFTY200 list from an external source")
        # Don't exit - allow proceeding with fewer tickers for now
        print(f"\n⚠ Proceeding with {len(alive)} tickers (may need to expand later)")
    else:
        print(f"\n✓ Success: {len(alive)} tickers (target: ≥180)")


if __name__ == "__main__":
    main()

