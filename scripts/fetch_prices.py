"""CLI to fetch historical prices via yfinance and save CSV for conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

from intentflow_ai.data.fetch_api import PriceFetchConfig, fetch_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NSE prices via yfinance.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--universe", required=True, help="Universe CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--suffix", default=".NS", help="Ticker suffix (default .NS)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PriceFetchConfig(
        start=args.start,
        end=args.end,
        suffix=args.suffix,
        universe_path=Path(args.universe),
        output_csv=Path(args.output),
    )
    fetch_and_save(cfg)


if __name__ == "__main__":
    main()
