"""
Ingest 10+ years of fundamental data from EOD Historical Data (EODHD) for NIFTY 500.
"""
import requests
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
from niftystocks import ns

API_KEY = "69217db2f2ae65.02354994"
BASE_URL = "https://eodhd.com/api/fundamentals"
RAW_DIR = Path("data/raw/eodhd")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def get_current_tickers():
    """Get list of current NIFTY 100 tickers we have price data for."""
    try:
        # Load from our local universe file
        universe_path = Path("data/external/universe/nifty100_universe.csv")
        df = pd.read_csv(universe_path)
        tickers = df['ticker'].dropna().tolist()
        print(f"✅ Loaded {len(tickers)} current tickers from local file")
        return tickers
    except Exception as e:
        print(f"❌ Failed to load tickers: {e}")
        return []

def fetch_fundamentals(ticker):
    """Fetch fundamentals for a single ticker."""
    # Check cache first
    cache_file = RAW_DIR / f"{ticker}.json"
    if cache_file.exists():
        # Check if cache is recent (e.g., < 1 week)
        # For now, just use it to save API calls during dev
        with open(cache_file, 'r') as f:
            return json.load(f)
            
    eod_ticker = f"{ticker}.NSE"
    url = f"{BASE_URL}/{eod_ticker}?api_token={API_KEY}&fmt=json"
    
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        # Save raw JSON
        with open(cache_file, 'w') as f:
            json.dump(data, f)
            
        return data
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None

def parse_fundamentals(ticker, data):
    """Parse EODHD JSON into a flat list of quarterly records."""
    if not data or 'Financials' not in data:
        return []
        
    financials = data['Financials']
    records = {}  # Key: date, Value: dict of metrics
    
    # Helper to process a statement type (Balance_Sheet, Income_Statement, Cash_Flow)
    def process_statement(statement_name):
        stmt = financials.get(statement_name, {}).get('quarterly', {})
        for date_str, values in stmt.items():
            if date_str not in records:
                records[date_str] = {'ticker': ticker, 'date': date_str, 'report_date': date_str}
            
            # Map fields to our schema
            # Note: EODHD field names are usually consistent
            for key, val in values.items():
                if val is not None:
                    try:
                        records[date_str][f"{statement_name.lower()}__{key}"] = float(val)
                    except:
                        pass

    process_statement('Balance_Sheet')
    process_statement('Income_Statement')
    process_statement('Cash_Flow')
    
    # Convert to list
    return list(records.values())

def main():
    print("=" * 60)
    print("EODHD Fundamental Data Ingestion")
    print("=" * 60)
    
    tickers = get_current_tickers()
    if not tickers:
        print("No tickers found!")
        return

    all_records = []
    
    print(f"\n[1/2] Fetching data for {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}", end="\r")
        
        data = fetch_fundamentals(ticker)
        if data:
            parsed = parse_fundamentals(ticker, data)
            all_records.extend(parsed)
            print(f"[{i}/{len(tickers)}] {ticker} ✅ {len(parsed)} qtrs")
        
        # Rate limit (EODHD is generous, but let's be safe)
        time.sleep(0.5)
        
    print(f"\n[2/2] Processing {len(all_records)} records...")
    df = pd.DataFrame(all_records)
    
    # Standardize columns
    # Map EODHD columns to our internal schema
    # This is a simplified mapping; we can add more later
    column_map = {
        'income_statement__totalRevenue': 'revenue',
        'income_statement__netIncome': 'net_income',
        'income_statement__ebitda': 'ebitda',
        'income_statement__eps': 'eps',
        'balance_sheet__totalAssets': 'total_assets',
        'balance_sheet__totalLiab': 'total_liabilities',
        'balance_sheet__totalStockholderEquity': 'total_equity',
        'cash_flow__totalCashFromOperatingActivities': 'operating_cash_flow',
        'cash_flow__capitalExpenditures': 'capex',
    }
    
    df = df.rename(columns=column_map)
    
    # Ensure dates
    df['date'] = pd.to_datetime(df['date'])
    df['report_date'] = pd.to_datetime(df['report_date'])
    
    # Add available_date (Reporting Delay)
    # EODHD gives filing date sometimes, but for now assume 45 days delay if not present
    # Ideally we use 'filing_date' if available
    df['available_date'] = df['report_date'] + pd.Timedelta(days=45)
    
    # Save
    output_path = PROCESSED_DIR / "fundamentals_eodhd.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"Total records: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
