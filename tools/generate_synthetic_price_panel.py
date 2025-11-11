"""Generate a synthetic multi-ticker price panel for local testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


TARGET_TICKERS = 200


def load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found at {path}")
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("ticker_nse") or cols.get("symbol")
    sector_col = cols.get("sector")
    if not ticker_col or not sector_col:
        raise ValueError(f"Universe file must contain ticker and sector columns. Found: {df.columns}")
    universe = df[[ticker_col, sector_col]].rename(columns={ticker_col: "ticker", sector_col: "sector"})
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    universe["sector"] = universe["sector"].astype(str).str.strip()
    universe = universe.drop_duplicates("ticker")
    if len(universe) >= TARGET_TICKERS:
        return universe

    needed = TARGET_TICKERS - len(universe)
    reps = []
    idx = 0
    while needed > 0:
        row = universe.iloc[idx % len(universe)].copy()
        row["ticker"] = f"{row['ticker']}_S{idx}"
        reps.append(row)
        idx += 1
        needed -= 1
    return pd.concat([universe, pd.DataFrame(reps)], ignore_index=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    universe_path = repo_root / "data" / "external" / "universe" / "nifty200.csv"
    universe = load_universe(universe_path)

    dates = pd.bdate_range("2018-01-01", "2025-10-31", freq="C")
    rng = np.random.default_rng(42)
    frames = []

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        sector = row["sector"]
        n = len(dates)
        rets = rng.normal(0.0002, 0.02, size=n)
        price = 100 * np.exp(np.cumsum(rets))
        opens = price * (1 + rng.normal(0, 0.002, size=n))
        highs = price * (1 + rng.uniform(0.001, 0.01, size=n))
        lows = price * (1 - rng.uniform(0.001, 0.01, size=n))
        vols = rng.integers(1e5, 5e6, size=n)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": price,
                    "volume": vols,
                    "sector": sector,
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    out_path = repo_root / "data" / "raw" / "price_confirmation" / "data.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    print("✅ synthetic data →", out_path)
    print(
        "rows:", len(panel),
        "| tickers:", panel["ticker"].nunique(),
        "| span:", panel["date"].min().date(), "→", panel["date"].max().date(),
    )


if __name__ == "__main__":
    main()
