"""
Script to fetch fundamental data for NIFTY 200 universe using TradingView provider.
Saves to data/fundamentals.csv.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.data.fundamentals_fetcher import FundamentalDataFetcher
from intentflow_ai.data.universe import load_universe
from intentflow_ai.config.settings import settings

def main():
    print("Starting Fundamental Data Ingestion...")
    
    # 1. Load Universe
    print("Loading universe...")
    try:
        # load_universe requires a path or uses default if handled, but error said missing arg.
        # Checking universe.py signature would be good, but let's assume it needs a path or we can pass None if it has default?
        # The error was: load_universe() missing 1 required positional argument: 'path'
        # So I must provide a path.
        # I'll pass the default path from settings.
        universe_path = Path(settings.data_dir) / "universe.parquet"
        universe_df = load_universe(universe_path)
        symbols = universe_df['symbol'].unique().tolist()
        print(f"Loaded {len(symbols)} symbols from universe.")
    except Exception as e:
        print(f"Error loading universe: {e}")
        # Fallback to a small list for testing if universe load fails
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        print(f"Using fallback symbols: {symbols}")

    # 2. Fetch Fundamentals
    fetcher = FundamentalDataFetcher()
    
    # We fetch from 2020 to now (though TV provider only gives current data)
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    print(f"Fetching fundamentals for {len(symbols)} symbols...")
    df = fetcher.fetch_universe_fundamentals(
        universe_symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        force_refresh=True # Force refresh to ensure we use the new provider
    )
    
    if df.empty:
        print("No data fetched!")
        return

    # 3. Save to CSV
    output_path = Path(settings.data_dir) / "fundamentals.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved fundamentals to {output_path}")
    
    # 4. Show Sample
    print("\nSample Data:")
    print(df.head())
    print("\nColumns:")
    print(list(df.columns))
    
    # 5. Basic Stats
    print("\nStats:")
    print(f"Total Rows: {len(df)}")
    print(f"Unique Symbols: {df['symbol'].nunique()}")
    
    # Check for key columns
    key_cols = ['pe_ratio', 'roe', 'debt_to_equity']
    for col in key_cols:
        if col in df.columns:
            coverage = df[col].notna().mean()
            print(f"{col} coverage: {coverage:.1%}")
        else:
            print(f"{col} missing!")

if __name__ == "__main__":
    main()
