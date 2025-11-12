import glob
import os
import sys
from pathlib import Path

import pandas as pd


def load_bhav_folder(folder: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        need = ["symbol", "timestamp", "open", "high", "low", "close", "tottrdqty"]
        if not all(k in cols for k in need):
            continue
        dd = pd.DataFrame(
            {
                "date": pd.to_datetime(df[cols["timestamp"]]),
                "ticker": df[cols["symbol"]].astype(str).str.strip().str.upper(),
                "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
                "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
                "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
                "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
                "volume": pd.to_numeric(df[cols["tottrdqty"]], errors="coerce"),
            }
        )
        frames.append(dd)
    if not frames:
        raise SystemExit("No usable bhavcopy CSVs found in folder.")
    return pd.concat(frames, ignore_index=True)


def load_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}
    req = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if not all(k in lc for k in req):
        raise SystemExit("Wide CSV missing required columns.")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[lc["date"]]),
            "ticker": df[lc["ticker"]].astype(str).str.strip().str.upper(),
            "open": pd.to_numeric(df[lc["open"]], errors="coerce"),
            "high": pd.to_numeric(df[lc["high"]], errors="coerce"),
            "low": pd.to_numeric(df[lc["low"]], errors="coerce"),
            "close": pd.to_numeric(df[lc["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[lc["volume"]], errors="coerce"),
        }
    )
    return out


def attach_pit_sector(px: pd.DataFrame, hist_path: str) -> pd.DataFrame:
    hist = pd.read_csv(hist_path)
    cols = {c.lower(): c for c in hist.columns}
    ticker_col = cols.get("ticker") or cols.get("ticker_nse") or cols.get("symbol")
    sector_col = cols.get("sector")
    start_col = cols.get("effective_from") or cols.get("start_date")
    end_col = cols.get("effective_to") or cols.get("end_date")
    if not ticker_col or not sector_col:
        raise ValueError("History CSV must contain ticker and sector columns.")

    hist = hist.rename(
        columns={
            ticker_col: "ticker",
            sector_col: "sector",
            start_col or "start_date": "start_date",
            end_col or "end_date": "end_date",
        }
    )
    hist["ticker"] = hist["ticker"].astype(str).str.strip().str.upper()
    hist["start_date"] = pd.to_datetime(hist["start_date"])
    hist["end_date"] = pd.to_datetime(hist["end_date"]).fillna(pd.Timestamp.max)

    px = px.sort_values(["ticker", "date"])
    px["ticker"] = px["ticker"].astype(str).str.upper().str.strip()

    merged = px.merge(hist, on="ticker", how="left")
    mask = (merged["date"] >= merged["start_date"]) & (merged["date"] <= merged["end_date"])
    merged.loc[~mask, "sector"] = None
    merged["sector"] = merged.groupby("ticker")["sector"].ffill().bfill()
    merged = merged.dropna(subset=["sector"])
    merged = merged.drop_duplicates(subset=["date", "ticker"]).sort_values(["date", "ticker"])

    cols = ["date", "ticker", "open", "high", "low", "close", "volume", "sector"]
    return merged[cols]


def main():
    if len(sys.argv) < 2:
        print("Usage:\n  python tools/convert_bhavcopy_to_parquet.py <bhav_folder_or_csv> [nifty200_history.csv]")
        sys.exit(1)
    src = sys.argv[1]
    hist = sys.argv[2] if len(sys.argv) > 2 else "data/external/universe/nifty200_history.csv"

    if os.path.isdir(src):
        px = load_bhav_folder(src)
    else:
        px = load_wide_csv(src)

    px = attach_pit_sector(px, hist)
    if px["ticker"].nunique() < 50:
        raise SystemExit(
            f"Only {px['ticker'].nunique()} tickers fetched. "
            "Ensure your data source covers the full NIFTY200 universe."
        )
    else:
        print(f"Tickers in panel: {px['ticker'].nunique()} (target 180)")

    out_dir = Path("data/raw/price_confirmation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    px.to_parquet(out_path, index=False)
    print(f"Wrote {len(px):,} rows to {out_path}")
    print(f"Dates: {px['date'].min().date()} → {px['date'].max().date()} | Tickers: {px['ticker'].nunique()}")


if __name__ == "__main__":
    main()
