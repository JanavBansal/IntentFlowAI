"""
Fetch historical OHLCV data for all NIFTY 500 stocks using yfinance.
This will expand our price coverage to 500 tickers.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import time
from niftystocks import ns

# Get NIFTY 500 ticker list using niftystocks package
def get_nifty_tickers():
    """Fetch NIFTY 500 constituent list using niftystocks package."""
    try:
        # Get NIFTY 500 constituents
        nifty500 = ns.get_nifty500()
        tickers = nifty500['Symbol'].tolist()
        print(f"✅ Fetched {len(tickers)} NIFTY 500 tickers using niftystocks package")
        return tickers
    except Exception as e:
        print(f"❌ Failed to fetch NIFTY 500 list: {e}")
        print("Falling back to local NIFTY 100 list...")
        # Fallback to our existing NIFTY 100 list
        universe_path = Path("data/external/universe/nifty100_universe.csv")
        df = pd.read_csv(universe_path)
        tickers = df['ticker'].dropna().tolist()
        print(f"✅ Loaded {len(tickers)} NIFTY 100 tickers from local file")
        return tickers

def fetch_prices_for_ticker(ticker, start_date, end_date):
    """Fetch price data for a single ticker."""
    try:
        # Add .NS suffix for NSE
        symbol = f"{ticker}.NS"
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            print(f"  ⚠️  No data for {ticker}")
            return None
        
        # Reset index to get 'Date' as a column
        data = data.reset_index()
        
        # Handle multi-index columns (yfinance sometimes returns these)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data['ticker'] = ticker
        
        # Rename columns to lowercase
        data.columns = [str(c).lower() for c in data.columns]
        
        print(f"  ✅ {ticker}: {len(data)} days")
        return data
    
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None

def main():
    print("=" * 60)
    print("NIFTY 500 Price Data Ingestion")
    print("=" * 60)
    
    # Configuration
    start_date = "2010-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    output_path = Path("data/processed/prices.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get tickers
    print("\n[1/3] Fetching ticker list...")
    tickers = get_nifty_tickers()
    
    # Fetch prices
    print(f"\n[2/3] Fetching price data for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}")
    print("This will take ~10-15 minutes (rate-limited to avoid bans)...\n")
    
    all_data = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        data = fetch_prices_for_ticker(ticker, start_date, end_date)
        
        if data is not None:
            all_data.append(data)
        
        # Rate limiting: 1 request every 2 seconds
        if i < len(tickers):
            time.sleep(2)
    
    # Combine and save
    print(f"\n[3/3] Saving to {output_path}...")
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.sort_values(['ticker', 'date'])
        
        # Save as parquet
        combined.to_parquet(output_path, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"Total rows: {len(combined):,}")
        print(f"Tickers with data: {combined['ticker'].nunique()}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Saved to: {output_path}")
    else:
        print("\n❌ No data fetched!")

if __name__ == "__main__":
    main()
