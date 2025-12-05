"""Fetch real sector data for the universe using yfinance.
Fixes the 'Unknown' sector issue in the dashboard.
"""
import pandas as pd
import yfinance as yf
import time
from pathlib import Path


def fetch_sectors():
    # 1. Load Universe
    universe_path = Path("data/external/universe/nifty464_universe.csv")
    if not universe_path.exists():
        print(f"❌ Universe file not found: {universe_path}")
        return
    
    df = pd.read_csv(universe_path)
    tickers = df['ticker'].tolist()
    print(f"🔍 Found {len(tickers)} tickers. Fetching sectors...")
    
    sector_map = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        try:
            # Add .NS for Yahoo
            yf_ticker = f"{ticker}.NS"
            info = yf.Ticker(yf_ticker).info
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Fallback for common missing ones
            if sector == 'Unknown':
                if 'BANK' in ticker: sector = 'Financial Services'
                elif 'PHARMA' in ticker: sector = 'Healthcare'
                elif 'INFY' in ticker or 'TCS' in ticker: sector = 'Technology'
            
            print(f"-> {sector}")
            sector_map.append({
                'ticker_nse': ticker,
                'sector': sector,
                'industry': industry
            })
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            sector_map.append({'ticker_nse': ticker, 'sector': 'Unknown', 'industry': 'Unknown'})
        
        # Be nice to API
        time.sleep(0.2)
    
    # 2. Save to static map
    result_df = pd.DataFrame(sector_map)
    output_path = Path("data/static/sector_map.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved sector map to {output_path}")
    
    # 3. Update Universe File (Optional but recommended)
    # Merge sector back into universe file if you want
    df_merged = df.merge(result_df[['ticker_nse', 'sector']], left_on='ticker', right_on='ticker_nse', how='left')
    if 'ticker_nse' in df_merged.columns:
        df_merged = df_merged.drop(columns=['ticker_nse'])
    
    # Save backup
    df.to_csv(str(universe_path) + ".bak", index=False)
    df_merged.to_csv(universe_path, index=False)
    print(f"✅ Updated universe file with sectors: {universe_path}")


if __name__ == "__main__":
    fetch_sectors()
