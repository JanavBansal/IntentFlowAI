"""
Debug TradingView provider.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.data.tradingview_provider import get_tradingview_provider
from datetime import datetime

def main():
    print("Debugging TradingView Provider...")
    provider = get_tradingview_provider()
    
    if not provider.is_available():
        print("Provider not available (library missing?)")
        return

    symbol = "RELIANCE"
    print(f"Fetching {symbol}...")
    
    try:
        df = provider.fetch_fundamentals(symbol, datetime(2020,1,1), datetime.now())
        if df.empty:
            print("Returned empty DataFrame.")
        else:
            print("Success!")
            print(df)
            print(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
