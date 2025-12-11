"""
IntentFlow V4.5 - NSE Delivery Data Fetcher
============================================
Fetch delivery volume data for Flow Detective agent.

Methods (ranked by speed):
1. nse library (deliveryBhavcopy) - FASTEST, includes DELIV_QTY and DELIV_PER
2. jugaad-data (bhavcopy) - Reliable for bulk downloads
3. Direct NSE Archive scraping - Backup method

Usage:
    python scripts/fetch_nse_delivery_data.py --method nse --start 2024-01-01
    python scripts/fetch_nse_delivery_data.py --method jugaad --start 2024-01-01
    python scripts/fetch_nse_delivery_data.py --method direct --start 2024-06-01
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import time
from random import randint
import warnings
warnings.filterwarnings('ignore')


# NIFTY 50 symbols (high liquidity)
NIFTY_50 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
    'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK',
    'ASIANPAINT', 'MARUTI', 'HCLTECH', 'SUNPHARMA', 'BAJFINANCE',
    'TITAN', 'WIPRO', 'ULTRACEMCO', 'NTPC', 'NESTLEIND', 'POWERGRID',
    'TECHM', 'M&M', 'TATAMOTORS', 'ADANIENT', 'ADANIPORTS', 'JSWSTEEL',
    'TATASTEEL', 'ONGC', 'COALINDIA', 'BAJAJFINSV', 'GRASIM', 'BRITANNIA',
    'CIPLA', 'DIVISLAB', 'APOLLOHOSP', 'EICHERMOT', 'HINDALCO',
    'INDUSINDBK', 'DRREDDY', 'TATACONSUM', 'BPCL', 'HEROMOTOCO',
    'SHRIRAMFIN', 'SBILIFE', 'HDFCLIFE', 'BAJAJ-AUTO', 'LTIM'
]


def fetch_with_nse_library(start_date: date, end_date: date, output_dir: str):
    """
    Use the 'nse' PyPI package - has dedicated deliveryBhavcopy!
    Install: pip install nse
    """
    try:
        from nse import NSE
    except ImportError:
        print("❌ 'nse' package not installed. Run: pip install nse")
        return None
    
    print("🚀 METHOD 1: Using NSE library (deliveryBhavcopy)")
    print(f"   Date range: {start_date} to {end_date}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_delivery_data = []
    
    with NSE(str(output_path)) as nse:
        current_date = start_date
        days_processed = 0
        
        while current_date <= end_date:
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            try:
                dlv_file = nse.deliveryBhavcopy(date=datetime.combine(current_date, datetime.min.time()))
                
                if dlv_file and os.path.exists(dlv_file):
                    df = pd.read_csv(dlv_file)
                    df['DATE'] = current_date
                    all_delivery_data.append(df)
                    days_processed += 1
                    print(f"   ✅ {current_date}: {len(df)} records")
                
                time.sleep(randint(1, 3))
                
            except Exception as e:
                print(f"   ⚠️ {current_date}: {str(e)[:50]}")
            
            current_date += timedelta(days=1)
            
            # Progress every 10 days
            if days_processed > 0 and days_processed % 10 == 0:
                print(f"   📊 Progress: {days_processed} trading days processed")
    
    if all_delivery_data:
        combined_df = pd.concat(all_delivery_data, ignore_index=True)
        output_file = output_path / "delivery_data_nse.parquet"
        combined_df.to_parquet(output_file)
        print(f"\n✅ Saved {len(combined_df)} records to {output_file}")
        return combined_df
    
    return None


def fetch_with_jugaad_data(start_date: date, end_date: date, output_dir: str):
    """
    Use jugaad-data for stock-level delivery data.
    Install: pip install jugaad-data
    """
    try:
        from jugaad_data.nse import stock_df, bhavcopy_save
    except ImportError:
        print("❌ 'jugaad-data' not installed. Run: pip install jugaad-data")
        return None
    
    print("🚀 METHOD 2: Using jugaad-data library")
    print(f"   Date range: {start_date} to {end_date}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download bhavcopy files for 30 days
    print("\n   📥 Downloading bhavcopy files...")
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    for dt in date_range[:30]:
        try:
            bhavcopy_save(dt.date(), str(output_path))
            print(f"   ✅ Bhavcopy: {dt.date()}")
            time.sleep(randint(1, 3))
        except Exception as e:
            print(f"   ⚠️ {dt.date()}: {str(e)[:30]}")
    
    # Fetch stock-level data
    print("\n   📥 Downloading stock data for NIFTY 50...")
    all_stock_data = []
    
    for symbol in NIFTY_50[:20]:
        try:
            df = stock_df(
                symbol=symbol,
                from_date=start_date,
                to_date=end_date,
                series="EQ"
            )
            df['SYMBOL'] = symbol
            all_stock_data.append(df)
            print(f"   ✅ {symbol}: {len(df)} days")
            time.sleep(randint(1, 2))
        except Exception as e:
            print(f"   ⚠️ {symbol}: {str(e)[:30]}")
    
    if all_stock_data:
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        output_file = output_path / "stock_data_jugaad.parquet"
        combined_df.to_parquet(output_file)
        print(f"\n✅ Saved {len(combined_df)} records to {output_file}")
        return combined_df
    
    return None


def fetch_from_nse_direct(start_date: date, end_date: date, output_dir: str):
    """
    Direct scraping from NSE archives.
    MTO URL: https://nsearchives.nseindia.com/archives/equities/mto/MTO_{DDMMYYYY}.DAT
    """
    import requests
    
    print("🚀 METHOD 3: Direct NSE Archive Scraping")
    print(f"   Date range: {start_date} to {end_date}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Referer': 'https://www.nseindia.com/'
    }
    
    session = requests.Session()
    session.get("https://www.nseindia.com/", headers=headers, timeout=10)
    
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        date_str = current_date.strftime("%d%m%Y")
        mto_url = f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{date_str}.DAT"
        
        try:
            response = session.get(mto_url, headers=headers, timeout=15)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                records = []
                for line in lines[4:]:
                    if len(line) > 50:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 4:
                                records.append({
                                    'DATE': current_date,
                                    'SYMBOL': parts[2].strip(),
                                    'DELIV_QTY': float(parts[3]) if parts[3].strip() else 0,
                                    'DELIV_PER': float(parts[4]) if len(parts) > 4 else 0
                                })
                        except:
                            continue
                
                if records:
                    all_data.extend(records)
                    print(f"   ✅ {current_date}: {len(records)} records")
        except Exception as e:
            print(f"   ❌ {current_date}: {str(e)[:30]}")
        
        time.sleep(randint(2, 5))
        current_date += timedelta(days=1)
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = output_path / "delivery_data_direct.parquet"
        df.to_parquet(output_file)
        print(f"\n✅ Saved {len(df)} records to {output_file}")
        return df
    
    return None


def compute_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute delivery features for Flow Detective agent."""
    print("\n📊 Computing delivery features...")
    
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    df = df.sort_values(['SYMBOL', 'DATE'])
    features = []
    
    for symbol, group in df.groupby('SYMBOL'):
        g = group.copy()
        
        # Delivery momentum
        g['delivery_pct_3d_change'] = g['DELIV_PER'].pct_change(3)
        g['delivery_pct_7d_change'] = g['DELIV_PER'].pct_change(7)
        
        # Delivery z-score (spike detection)
        rolling_mean = g['DELIV_QTY'].rolling(20).mean()
        rolling_std = g['DELIV_QTY'].rolling(20).std()
        g['deliv_qty_zscore'] = (g['DELIV_QTY'] - rolling_mean) / rolling_std
        
        # Large delivery flag
        g['large_delivery_flag'] = (g['DELIV_QTY'] > g['DELIV_QTY'].quantile(0.95)).astype(int)
        
        # Moving averages
        g['deliv_ratio_ma_5'] = g['DELIV_PER'].rolling(5).mean()
        g['deliv_ratio_ma_20'] = g['DELIV_PER'].rolling(20).mean()
        
        features.append(g)
    
    result = pd.concat(features, ignore_index=True)
    print(f"   ✅ Features computed for {df['SYMBOL'].nunique()} symbols")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Fetch NSE delivery data for Flow Detective')
    parser.add_argument('--method', choices=['nse', 'jugaad', 'direct', 'all'], default='jugaad')
    parser.add_argument('--start', type=str, default='2024-01-01')
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--output', type=str, default='./data/raw/delivery_transactions/')
    parser.add_argument('--compute-features', action='store_true', default=True)
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else date.today()
    
    print("=" * 60)
    print("IntentFlow V4.5 - NSE Delivery Data Fetcher")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    df = None
    
    if args.method == 'nse':
        df = fetch_with_nse_library(start_date, end_date, args.output)
    elif args.method == 'jugaad':
        df = fetch_with_jugaad_data(start_date, end_date, args.output)
    elif args.method == 'direct':
        df = fetch_from_nse_direct(start_date, end_date, args.output)
    elif args.method == 'all':
        df = fetch_with_nse_library(start_date, end_date, args.output)
        if df is None:
            df = fetch_with_jugaad_data(start_date, end_date, args.output)
        if df is None:
            df = fetch_from_nse_direct(start_date, end_date, args.output)
    
    if df is not None and args.compute_features:
        df = compute_delivery_features(df)
        output_path = Path(args.output) / "delivery_features.parquet"
        df.to_parquet(output_path)
        print(f"\n✅ Saved features to {output_path}")
    
    print("\n" + "=" * 60)
    print("DONE! Next: Re-run test_council.py to verify Flow Detective")
    print("=" * 60)


if __name__ == "__main__":
    main()
