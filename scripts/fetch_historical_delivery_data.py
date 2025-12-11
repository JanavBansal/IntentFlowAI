#!/usr/bin/env python3
"""
Fetch historical NSE delivery data using jugaad-data.

This script backfills REAL delivery transaction data from NSE for V4.5 Council of Experts.
The Flow Detective agent depends on this data for institutional activity detection.

Usage:
    # Test with a few stocks first
    python scripts/fetch_historical_delivery_data.py --symbols RELIANCE,TCS,INFY --start 2024-01-01
    
    # Fetch for top 50 stocks
    python scripts/fetch_historical_delivery_data.py --top 50 --start 2015-01-01
    
    # Fetch all stocks (long running)
    python scripts/fetch_historical_delivery_data.py --all --start 2020-01-01
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import time
import sys

import pandas as pd


def get_ticker_list(n_tickers: int = None, specific_symbols: list = None) -> list:
    """Get list of NSE tickers from sector_map.csv."""
    sector_map_path = Path("data/static/sector_map.csv")
    
    if not sector_map_path.exists():
        print(f"❌ Sector map not found: {sector_map_path}")
        sys.exit(1)
    
    df = pd.read_csv(sector_map_path)
    all_tickers = df['ticker_nse'].dropna().tolist()
    
    if specific_symbols:
        # Filter to specific symbols
        tickers = [t for t in all_tickers if t in specific_symbols]
        missing = set(specific_symbols) - set(tickers)
        if missing:
            print(f"⚠️ Symbols not in universe: {missing}")
        return tickers
    
    if n_tickers:
        return all_tickers[:n_tickers]
    
    return all_tickers


def fetch_delivery_data_jugaad(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch delivery data for a single symbol using jugaad-data."""
    try:
        from jugaad_data.nse import bhavcopy
    except ImportError:
        print("❌ jugaad-data not installed. Run: pip install jugaad-data")
        sys.exit(1)
    
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            # Fetch bhavcopy for the date (contains delivery data)
            df = bhavcopy(current_date)
            
            if df is not None and len(df) > 0:
                # Filter for our symbol
                symbol_data = df[df['SYMBOL'] == symbol]
                
                if len(symbol_data) > 0:
                    row = symbol_data.iloc[0]
                    
                    # Extract delivery columns
                    record = {
                        'date': current_date,
                        'ticker': symbol,
                        'open': row.get('OPEN_PRICE', row.get('OPEN', None)),
                        'high': row.get('HIGH_PRICE', row.get('HIGH', None)),
                        'low': row.get('LOW_PRICE', row.get('LOW', None)),
                        'close': row.get('CLOSE_PRICE', row.get('CLOSE', None)),
                        'volume': row.get('TTL_TRD_QNTY', row.get('TOTTRDQTY', 0)),
                        'delivery_qty': row.get('DELIV_QTY', row.get('DELIVERABLEQUANTITY', 0)),
                        'delivery_pct': row.get('DELIV_PER', row.get('DELIVERYPERCENTAGE', 0)),
                    }
                    all_data.append(record)
        except Exception as e:
            # Skip weekends/holidays silently
            if "No data" not in str(e) and "404" not in str(e):
                pass  # Silently skip errors for now
        
        current_date += timedelta(days=1)
    
    if all_data:
        return pd.DataFrame(all_data)
    return pd.DataFrame()


def fetch_delivery_data_alternative(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Alternative approach using stock_df with delivery."""
    try:
        from jugaad_data.nse import stock_df
    except ImportError:
        print("❌ jugaad-data not installed. Run: pip install jugaad-data")
        sys.exit(1)
    
    try:
        df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date, series="EQ")
        
        if df is not None and len(df) > 0:
            # Rename columns to our format
            df = df.rename(columns={
                'DATE': 'date',
                'SYMBOL': 'ticker', 
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'TOTTRDQTY': 'volume',
                'DELIVERABLEQUANTITY': 'delivery_qty',
                'DELIVERYPERCENTAGE': 'delivery_pct',
                '%DELTO TRADED QTY': 'delivery_pct',
                'DELIV_QTY': 'delivery_qty',
                'DELIV_PER': 'delivery_pct',
            })
            
            df['ticker'] = symbol
            
            # Keep only needed columns
            cols_to_keep = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'delivery_qty', 'delivery_pct']
            available_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[available_cols]
            
            return df
    except Exception as e:
        print(f"  ⚠️ {symbol}: {e}")
    
    return pd.DataFrame()


def backfill_delivery_data(
    symbols: list,
    start_date: date,
    end_date: date,
    output_dir: Path,
    delay_seconds: float = 0.5
) -> dict:
    """
    Fetch REAL NSE delivery data for all stocks.
    
    Returns:
        dict with 'success' and 'failed' lists
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'success': [], 'failed': []}
    
    print(f"\n🚀 Starting delivery data fetch for {len(symbols)} symbols")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Output dir: {output_dir}")
    print("-" * 60)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Fetching {symbol}...", end=" ")
        
        try:
            # Try alternative method first (usually more reliable)
            df = fetch_delivery_data_alternative(symbol, start_date, end_date)
            
            if df is not None and len(df) > 0:
                # Save to parquet
                output_file = output_dir / f"{symbol}_delivery.parquet"
                df.to_parquet(output_file, index=False)
                
                print(f"✅ {len(df)} rows")
                results['success'].append(symbol)
            else:
                print("⚠️ No data")
                results['failed'].append(symbol)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results['failed'].append(symbol)
        
        # Rate limiting
        time.sleep(delay_seconds)
    
    return results


def consolidate_delivery_files(output_dir: Path) -> pd.DataFrame:
    """Consolidate individual stock files into single parquet for easier loading."""
    all_files = list(output_dir.glob("*_delivery.parquet"))
    
    if not all_files:
        print("No delivery files found to consolidate")
        return pd.DataFrame()
    
    print(f"\n📦 Consolidating {len(all_files)} delivery files...")
    
    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Failed to read {f.name}: {e}")
    
    if dfs:
        consolidated = pd.concat(dfs, ignore_index=True)
        
        # Save consolidated file
        consolidated_path = output_dir / "all_delivery.parquet"
        consolidated.to_parquet(consolidated_path, index=False)
        
        print(f"✅ Consolidated {len(consolidated)} total rows")
        print(f"   Saved to: {consolidated_path}")
        
        return consolidated
    
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Fetch NSE delivery data using jugaad-data")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols (e.g., RELIANCE,TCS,INFY)")
    parser.add_argument("--top", type=int, help="Fetch top N stocks from universe")
    parser.add_argument("--all", action="store_true", help="Fetch all 464 stocks")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--output", type=str, default="data/raw/delivery_transactions/", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    parser.add_argument("--consolidate-only", action="store_true", help="Only consolidate existing files")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Handle consolidate-only mode
    if args.consolidate_only:
        consolidate_delivery_files(output_dir)
        return
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    
    # Get ticker list
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        tickers = get_ticker_list(specific_symbols=symbols)
    elif args.top:
        tickers = get_ticker_list(n_tickers=args.top)
    elif args.all:
        tickers = get_ticker_list()
    else:
        # Default: top 10 for quick testing
        tickers = get_ticker_list(n_tickers=10)
        print("⚠️ No --symbols, --top, or --all specified. Using top 10 stocks for testing.")
    
    # Fetch data
    results = backfill_delivery_data(
        symbols=tickers,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        delay_seconds=args.delay
    )
    
    # Consolidate
    consolidated_df = consolidate_delivery_files(output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    print(f"✅ Success: {len(results['success'])} stocks")
    print(f"❌ Failed: {len(results['failed'])} stocks")
    
    if results['failed']:
        print(f"\nFailed symbols: {', '.join(results['failed'][:20])}")
        if len(results['failed']) > 20:
            print(f"  ... and {len(results['failed']) - 20} more")
    
    if len(consolidated_df) > 0:
        print(f"\n📊 Consolidated data:")
        print(f"   Total rows: {len(consolidated_df):,}")
        print(f"   Unique tickers: {consolidated_df['ticker'].nunique()}")
        if 'date' in consolidated_df.columns:
            print(f"   Date range: {consolidated_df['date'].min()} to {consolidated_df['date'].max()}")
        
        # Check for delivery columns
        if 'delivery_qty' in consolidated_df.columns:
            non_null = consolidated_df['delivery_qty'].notna().sum()
            print(f"   Delivery qty non-null: {non_null:,} ({non_null/len(consolidated_df)*100:.1f}%)")
        if 'delivery_pct' in consolidated_df.columns:
            non_null = consolidated_df['delivery_pct'].notna().sum()
            print(f"   Delivery pct non-null: {non_null:,} ({non_null/len(consolidated_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
