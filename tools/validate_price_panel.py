import pathlib

import pandas as pd

p = pathlib.Path("data/raw/price_confirmation/data.parquet")
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
print("rows:", len(df))
print("unique tickers:", tickers)
print("date range:", dmin.date(), "→", dmax.date())
assert tickers >= 180, "Need ≥180 tickers for NIFTY200 coverage"
assert pd.Timestamp("2023-07-01") >= dmin, "Start must be ≤ 2023-07-01 (train window start)"

blank_sector = df["sector"].isna().sum() + (df["sector"].astype(str).str.strip() == "").sum()
assert blank_sector == 0, f"{blank_sector} rows missing sector"

df.sort_values(["date", "ticker"], inplace=True)
df.to_parquet(p, index=False)
print("OK ✔ panel looks good.")
