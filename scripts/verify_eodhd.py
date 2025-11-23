"""
Verify EODHD API key and fetch sample fundamental data.
"""
import requests
import json

API_KEY = "69217db2f2ae65.02354994"
TICKER = "RELIANCE.NSE"  # EODHD uses .NSE suffix

def verify_api():
    url = f"https://eodhd.com/api/fundamentals/{TICKER}?api_token={API_KEY}&fmt=json"
    
    print(f"Testing API for {TICKER}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print("✅ API Connection Successful!")
        print(f"Name: {data['General']['Name']}")
        print(f"Sector: {data['General']['Sector']}")
        
        # Check for Financials
        financials = data.get('Financials', {})
        bs = financials.get('Balance_Sheet', {}).get('quarterly', {})
        print(f"✅ Found {len(bs)} quarterly balance sheet records")
        
        # Print first record date
        dates = sorted(bs.keys(), reverse=True)
        if dates:
            print(f"Latest Report: {dates[0]}")
            print(f"Oldest Report: {dates[-1]}")
            
    except Exception as e:
        print(f"❌ API Failed: {e}")
        print(response.text if 'response' in locals() else "")

if __name__ == "__main__":
    verify_api()
