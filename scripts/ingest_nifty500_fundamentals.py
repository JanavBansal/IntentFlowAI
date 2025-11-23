"""
Merge NIFTY 500 list with EODHD NSE tickers to create the master universe.
Then ingest fundamentals for all tickers.
"""
import pandas as pd
import requests
import json
import time
from pathlib import Path

API_KEY = "69217db2f2ae65.02354994"
BASE_URL = "https://eodhd.com/api/fundamentals"
RAW_DIR = Path("data/raw/eodhd")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load lists
nifty500 = pd.read_csv("data/external/universe/nifty500_constituents.csv")
eodhd_nse = pd.read_csv("data/external/universe/eodhd_nse_tickers.csv")

print(f"NIFTY 500: {len(nifty500)} tickers")
print(f"EODHD NSE: {len(eodhd_nse)} tickers")

# Check if we already have fundamentals
existing_file = PROCESSED_DIR / "fundamentals_eodhd.parquet"
if existing_file.exists():
    existing_df = pd.read_parquet(existing_file)
    existing_tickers = set(existing_df['ticker'].unique())
    print(f"Already have fundamentals for {len(existing_tickers)} tickers")
else:
    existing_tickers = set()

# Merge: Find NIFTY 500 tickers in EODHD list
# The NIFTY 500 list has 'Symbol' column, EODHD has 'Code' column
nifty_symbols = set(nifty500['Symbol'].str.upper())
eodhd_nse['Code'] = eodhd_nse['Code'].str.upper()

# Match by code (ticker symbol)
matched = eodhd_nse[eodhd_nse['Code'].isin(nifty_symbols)].copy()
print(f"Matched {len(matched)} tickers between NIFTY 500 and EODHD")

# Filter to new tickers only
new_tickers = matched[~matched['Code'].isin(existing_tickers)]
print(f"Need to fetch {len(new_tickers)} new tickers")

# If all tickers already exist, skip
if len(new_tickers) == 0:
    print("✅ All NIFTY 500 fundamentals already ingested!")
    exit(0)

# Fetch fundamentals for new tickers
def fetch_fundamentals(ticker):
    cache_file = RAW_DIR / f"{ticker}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
            
    eod_ticker = f"{ticker}.NSE"
    url = f"{BASE_URL}/{eod_ticker}?api_token={API_KEY}&fmt=json"
    
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
            
        return data
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None

def parse_fundamentals(ticker, data):
    if not data or 'Financials' not in data:
        return []
        
    financials = data['Financials']
    records = {}
    
    def process_statement(statement_name):
        stmt = financials.get(statement_name, {}).get('quarterly', {})
        for date_str, values in stmt.items():
            if date_str not in records:
                records[date_str] = {'ticker': ticker, 'date': date_str, 'report_date': date_str}
            
            for key, val in values.items():
                if val is not None:
                    try:
                        records[date_str][f"{statement_name.lower()}__{key}"] = float(val)
                    except:
                        pass

    process_statement('Balance_Sheet')
    process_statement('Income_Statement')
    process_statement('Cash_Flow')
    
    return list(records.values())

print(f"\nFetching fundamentals for {len(new_tickers)} new tickers...")
all_records = []

for idx, (_, row) in enumerate(new_tickers.iterrows(), 1):
    ticker = row['Code']
    print(f"[{idx}/{len(new_tickers)}] {ticker}", end="\r")
    
    data = fetch_fundamentals(ticker)
    if data:
        parsed = parse_fundamentals(ticker, data)
        all_records.extend(parsed)
        print(f"[{idx}/{len(new_tickers)}] {ticker} ✅ {len(parsed)} qtrs")
    
    time.sleep(0.5)

if len(all_records) == 0:
    print("\n❌ No new fundamental data fetched!")
    exit(0)

print(f"\nProcessing {len(all_records)} new records...")
new_df = pd.DataFrame(all_records)

# Save new data first (in case concat fails)
temp_file = PROCESSED_DIR / "fundamentals_new_batch.parquet"
print(f"Saving new batch to temporary file: {temp_file}")
new_df.to_parquet(temp_file, index=False)
print(f"✅ Saved {len(new_df):,} new records")

# Now try to merge with existing
try:
    if existing_file.exists():
        print("Merging with existing data...")
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Standardize columns
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

    combined = combined.rename(columns=column_map)
    combined['date'] = pd.to_datetime(combined['date'])
    combined['report_date'] = pd.to_datetime(combined['report_date'])
    combined['available_date'] = combined['report_date'] + pd.Timedelta(days=45)

    # Save
    combined.to_parquet(existing_file, index=False)

    print(f"\n✅ SUCCESS!")
    print(f"Total records: {len(combined):,}")
    print(f"Tickers: {combined['ticker'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Saved to: {existing_file}")
    
except Exception as e:
    print(f"\n❌ Error during merge/save: {e}")
    print(f"New data saved to: {temp_file}")
    print("You can manually merge later")
    import traceback
    traceback.print_exc()

