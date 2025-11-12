from __future__ import annotations

import argparse
import pathlib

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate price parquet coverage.")
    parser.add_argument("--path", default="data/raw/price_confirmation/data.parquet", help="Parquet file path.")
    parser.add_argument("--min-tickers", type=int, default=180, help="Minimum unique ticker count required.")
    parser.add_argument(
        "--min-span-years",
        type=float,
        default=5.0,
        help="Minimum historical span in years (approx) required.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p = pathlib.Path(args.path)
    assert p.exists(), f"Missing file: {p}"
    df = pd.read_parquet(p)

    required = ["date", "ticker", "open", "high", "low", "close", "volume", "sector"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    bad_vol = (df["volume"] < 0).sum()
    assert bad_price == 0, f"{bad_price} rows have non-positive prices"
    assert bad_vol == 0, f"{bad_vol} rows have negative volume"

    dupes = df.duplicated(["date", "ticker"]).sum()
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate (date,ticker) rows; keeping first")
        df = df.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"])

    tickers = df["ticker"].nunique()
    dmin, dmax = df["date"].min(), df["date"].max()
    span_years = (dmax - dmin).days / 365.25
    print("rows:", len(df))
    print("unique tickers:", tickers)
    print("date range:", dmin.date(), "→", dmax.date())
    print(f"span ~{span_years:.1f} years")
    assert tickers >= args.min_tickers, f"Need ≥{args.min_tickers} tickers for coverage"
    assert span_years >= args.min_span_years, f"Need ≥{args.min_span_years} years of history"

    blank_sector = df["sector"].isna().sum() + (df["sector"].astype(str).str.strip() == "").sum()
    assert blank_sector == 0, f"{blank_sector} rows missing sector"

    df.sort_values(["date", "ticker"], inplace=True)
    df.to_parquet(p, index=False)
    print("OK ✔ panel looks good.")


if __name__ == "__main__":
    main()
