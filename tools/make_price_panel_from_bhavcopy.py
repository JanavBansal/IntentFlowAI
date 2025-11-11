import sys
import os
import glob
import gzip
import io
from pathlib import Path

import numpy as np  # noqa: F401 - kept to mirror original instructions
import pandas as pd


def read_csv_any(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return pd.read_csv(io.BytesIO(fh.read()))
    return pd.read_csv(path)


def parse_date(value):
    try:
        return pd.to_datetime(value, errors="coerce").tz_localize(None)
    except Exception:
        return pd.NaT


assert len(sys.argv) >= 2, "Usage: python tools/make_price_panel_from_bhavcopy.py '<glob_or_dir>' [sector_csv]"
src_arg = os.path.expanduser(sys.argv[1])
sector_csv = os.path.expanduser(sys.argv[2]) if len(sys.argv) >= 3 else "data/static/sector_map.csv"

src_path = Path(src_arg)
if src_path.is_dir():
    files = sorted(list(src_path.glob("*.csv")) + list(src_path.glob("*.CSV")) + list(src_path.glob("*.csv.gz")))
else:
    files = sorted(glob.glob(src_arg))
assert files, f"No files matched {src_arg}"

dfs = []
for f in files:
    path = Path(f)
    df_raw = read_csv_any(path)
    upper_map = {col.upper(): col for col in df_raw.columns}
    rename = {}
    for src, dst in {
        "SYMBOL": "ticker",
        "TIMESTAMP": "date",
        "DATE": "date",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "TOTTRDQTY": "volume",
        "VOLUME": "volume",
    }.items():
        if src in upper_map:
            rename[upper_map[src]] = dst
    df_raw = df_raw.rename(columns=rename)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df_raw.columns)
    if missing:
        raise AssertionError(f"{path} missing columns: {sorted(missing)}")
    df = df_raw[list(required)].copy()
    df["date"] = df["date"].apply(parse_date)
    dfs.append(df)

panel = pd.concat(dfs, ignore_index=True)

uni_path = Path("data/static/nifty200_membership.csv")
if uni_path.exists():
    uni_df = pd.read_csv(uni_path)
    panel = panel[panel["ticker"].isin(set(uni_df["ticker"]))]

sector_map = Path(sector_csv)
if sector_map.exists():
    sectors = pd.read_csv(sector_map)
else:
    sectors = pd.DataFrame(columns=["ticker", "sector"])
if "sector" not in sectors.columns:
    sectors["sector"] = np.nan
panel = panel.merge(sectors[["ticker", "sector"]], on="ticker", how="left")

panel = panel.dropna(subset=["date", "ticker"])
for col in ["open", "high", "low", "close", "volume"]:
    panel[col] = pd.to_numeric(panel[col], errors="coerce")
panel = panel.dropna(subset=["open", "high", "low", "close", "volume"])
panel = panel[
    (panel["open"] > 0)
    & (panel["high"] > 0)
    & (panel["low"] > 0)
    & (panel["close"] > 0)
    & (panel["volume"] >= 0)
]
panel = panel.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"])

out_path = Path("data/raw/price_confirmation/data.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
panel.to_parquet(out_path, index=False)
print(
    "Wrote:",
    out_path,
    "| rows:",
    len(panel),
    "| tickers:",
    panel["ticker"].nunique(),
    "| dates:",
    panel["date"].min().date(),
    "→",
    panel["date"].max().date(),
)
