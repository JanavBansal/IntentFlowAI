"""
Fetch all NSE tickers from EODHD to build a robust universe.
"""
import requests
import pandas as pd
from pathlib import Path

API_KEY = "69217db2f2ae65.02354994"
URL = f"https://eodhd.com/api/exchange-symbol-list/NSE?api_token={API_KEY}&fmt=json"
OUTPUT_FILE = Path("data/external/universe/eodhd_nse_tickers.csv")

def fetch_nse_tickers():
    print(f"Fetching NSE ticker list from EODHD...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        print(f"✅ Fetched {len(df)} tickers from EODHD")
        
        # Filter for Common Stocks (Type 'Common Stock')
        if 'Type' in df.columns:
            df = df[df['Type'] == 'Common Stock']
            print(f"✅ Filtered to {len(df)} Common Stocks")
            
        # Save
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to fetch from EODHD: {e}")
        return None

if __name__ == "__main__":
    fetch_nse_tickers()
