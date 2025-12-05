"""Fetch real current stock prices from Yahoo Finance for NIFTY universe.
Replaces synthetic/mock data with real market data.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')


def fetch_real_prices():
    print("=" * 60)
    print("🚀 FETCHING REAL STOCK PRICES FROM YAHOO FINANCE")
    print("=" * 60)
    
    # 1. Load Universe
    universe_path = Path("data/external/universe/nifty464_universe.csv")
    if not universe_path.exists():
        print(f"❌ Universe file not found: {universe_path}")
        return None
    
    df = pd.read_csv(universe_path)
    tickers = df['ticker'].tolist()
    print(f"📊 Found {len(tickers)} tickers in universe")
    
    # 2. Define date range (last 5 years for good training data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    print(f"📅 Fetching data from {start_date.date()} to {end_date.date()}")
    
    # 3. Load sector map
    sector_map = {}
    sector_map_path = Path("data/static/sector_map.csv")
    if sector_map_path.exists():
        sector_df = pd.read_csv(sector_map_path)
        sector_map = dict(zip(sector_df["ticker_nse"], sector_df["sector"]))
    
    # 4. Fetch prices one by one (more reliable)
    all_data = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        yf_ticker = f"{ticker}.NS"
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        try:
            stock = yf.Ticker(yf_ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if data.empty:
                print("❌ No data")
                failed_tickers.append(ticker)
                continue
            
            data = data.reset_index()
            data.columns = [c.lower().replace(' ', '_') for c in data.columns]
            
            # Standardize columns
            ticker_data = pd.DataFrame({
                'date': pd.to_datetime(data['date']).dt.tz_localize(None),
                'ticker': ticker,
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
                'sector': sector_map.get(ticker, 'Unknown')
            })
            
            ticker_data = ticker_data.dropna(subset=['close'])
            
            if not ticker_data.empty:
                all_data.append(ticker_data)
                print(f"✅ {len(ticker_data)} rows")
            else:
                print("❌ Empty after filter")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            failed_tickers.append(ticker)
        
        # Rate limiting - be nice to Yahoo
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
    
    if not all_data:
        print("❌ No data fetched!")
        return None
    
    # 5. Combine all data
    print("\n📦 Combining data...")
    prices_df = pd.concat(all_data, ignore_index=True)
    prices_df = prices_df.sort_values(['ticker', 'date'])
    
    # 6. Summary stats
    print("\n" + "=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(prices_df):,}")
    print(f"Unique tickers: {prices_df['ticker'].nunique()}")
    print(f"Date range: {prices_df['date'].min().date()} to {prices_df['date'].max().date()}")
    print(f"Failed tickers: {len(failed_tickers)}")
    if failed_tickers[:10]:
        print(f"   First 10: {failed_tickers[:10]}")
    
    # Sector distribution
    print("\n📊 SECTOR DISTRIBUTION:")
    print(prices_df.groupby('sector')['ticker'].nunique().sort_values(ascending=False).head(10))
    
    # 7. Save to parquet
    output_path = Path("data/processed/prices.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_parquet(output_path, index=False)
    print(f"\n✅ Saved {len(prices_df):,} records to {output_path}")
    
    # 8. Also save to CSV for backup
    csv_path = Path("data/raw/price_confirmation/all_prices.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV backup to {csv_path}")
    
    return prices_df


if __name__ == "__main__":
    fetch_real_prices()
