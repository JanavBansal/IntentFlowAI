
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from intentflow_ai.data.nse_options_provider import NSEOptionsProvider

def test_scraper():
    print("Testing NSE Options Scraper...")
    provider = NSEOptionsProvider()
    
    # Test NIFTY (Index)
    print("\nFetching NIFTY Options Chain...")
    nifty_data = provider.get_pcr("NIFTY")
    print(f"NIFTY Data: {nifty_data}")
    
    if nifty_data.get("pcr") is not None and not isinstance(nifty_data.get("pcr"), float):
         print("✅ NIFTY PCR fetched successfully")
    elif isinstance(nifty_data.get("pcr"), float) and str(nifty_data.get("pcr")) != "nan":
         print("✅ NIFTY PCR fetched successfully")
    else:
         print("⚠️ NIFTY PCR is NaN (Market might be closed or API blocked)")

    # Test RELIANCE (Stock)
    print("\nFetching RELIANCE Options Chain...")
    stock_data = provider.get_pcr("RELIANCE")
    print(f"RELIANCE Data: {stock_data}")

if __name__ == "__main__":
    test_scraper()
