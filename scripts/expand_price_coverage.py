"""
Fetch historical OHLCV data for all tickers that have fundamentals but no prices.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import time

# Configuration
START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
PRICE_FILE = Path("data/processed/prices.parquet")
FUNDAMENTALS_FILE = Path("data/processed/fundamentals_eodhd.parquet")

def get_tickers_needing_prices():
    """Get list of tickers that have fundamentals but no price data."""
    # Load fundamentals
    fund_df = pd.read_parquet(FUNDAMENTALS_FILE)
    fund_tickers = set(fund_df['ticker'].unique())
    print(f"Tickers with fundamentals: {len(fund_tickers)}")
    
    # Load existing prices
    if PRICE_FILE.exists():
        price_df = pd.read_parquet(PRICE_FILE)
        price_tickers = set(price_df['ticker'].unique())
        print(f"Tickers with prices: {len(price_tickers)}")
    else:
        price_tickers = set()
        print("No existing price data found")
    
    # Find missing
    missing = sorted(fund_tickers - price_tickers)
    print(f"Tickers needing prices: {len(missing)}")
    
    return missing, price_df if PRICE_FILE.exists() else None

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
    print("Price Data Expansion")
    print("=" * 60)
    
    # Get tickers needing prices
    print("\n[1/3] Identifying tickers...")
    missing_tickers, existing_prices = get_tickers_needing_prices()
    
    if len(missing_tickers) == 0:
        print("✅ All tickers already have price data!")
        return
    
    # Fetch prices
    print(f"\n[2/3] Fetching price data for {len(missing_tickers)} tickers...")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("This will take ~2-3 hours with rate limiting...\n")
    
    all_data = []
    success_count = 0
    
    for i, ticker in enumerate(missing_tickers, 1):
        print(f"[{i}/{len(missing_tickers)}] {ticker}")
        data = fetch_prices_for_ticker(ticker, START_DATE, END_DATE)
        
        if data is not None:
            all_data.append(data)
            success_count += 1
        
        # Rate limiting: 1 request every 2 seconds to avoid bans
        if i < len(missing_tickers):
            time.sleep(2)
        
        # Save periodically (every 50 tickers)
        if i % 50 == 0 and len(all_data) > 0:
            print(f"\n💾 Checkpoint: Saving progress ({success_count} tickers fetched)...")
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df = temp_df.sort_values(['ticker', 'date'])
            
            # Merge with existing
            if existing_prices is not None:
                combined = pd.concat([existing_prices, temp_df], ignore_index=True)
            else:
                combined = temp_df
            
            # Save
            combined.to_parquet(PRICE_FILE, index=False)
            print(f"✅ Checkpoint saved: {combined['ticker'].nunique()} total tickers\n")
    
    # Final save
    print(f"\n[3/3] Final save...")
    if len(all_data) > 0:
        new_df = pd.concat(all_data, ignore_index=True)
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df = new_df.sort_values(['ticker', 'date'])
        
        # Merge with existing
        if existing_prices is not None:
            combined = pd.concat([existing_prices, new_df], ignore_index=True)
        else:
            combined = new_df
        
        # Save
        combined.to_parquet(PRICE_FILE, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"New tickers fetched: {success_count}/{len(missing_tickers)}")
        print(f"Total rows: {len(combined):,}")
        print(f"Total tickers: {combined['ticker'].nunique()}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Saved to: {PRICE_FILE}")
    else:
        print("\n❌ No new data fetched!")

if __name__ == "__main__":
    main()
