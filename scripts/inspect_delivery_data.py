"""Inspect current delivery and FII/DII data to validate contents."""

from pathlib import Path
import pandas as pd
import sys

def inspect_delivery_data():
    """Check what's in the delivery transactions directory."""
    
    print("=" * 60)
    print("DELIVERY DATA INSPECTION")
    print("=" * 60)
    
    delivery_dir = Path("data/raw/delivery_transactions")
    
    # List files
    print(f"\nDirectory: {delivery_dir}")
    if not delivery_dir.exists():
        print("❌ Directory does not exist!")
        return False
    
    files = list(delivery_dir.glob("*"))
    print(f"Files found: {len(files)}")
    for f in files[:10]:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    # Try to load parquet files
    parquet_files = list(delivery_dir.glob("*.parquet"))
    if parquet_files:
        print("\n--- Loading first parquet file ---")
        df = pd.read_parquet(parquet_files[0])
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSample data:")
        print(df.head(10).to_string())
        
        # Check for mock indicators
        print("\n--- Data Quality Check ---")
        if 'ticker' in df.columns:
            unique_tickers = df['ticker'].unique()
            print(f"Unique tickers: {len(unique_tickers)}")
            print(f"Sample tickers: {unique_tickers[:10]}")
            
            if "MOCK" in str(unique_tickers):
                print("❌ MOCK DATA DETECTED!")
                return False
        
        if 'delivery_ratio' in df.columns:
            print(f"\nDelivery ratio stats:")
            print(df['delivery_ratio'].describe())
            
            # Check if all values are the same (mock indicator)
            if df['delivery_ratio'].nunique() <= 3:
                print("⚠️ WARNING: Very few unique delivery_ratio values - likely mock data")
                return False
        
        if 'date' in df.columns:
            print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    
    return True


def inspect_fii_dii_data():
    """Check FII/DII data quality."""
    
    print("\n" + "=" * 60)
    print("FII/DII DATA INSPECTION")
    print("=" * 60)
    
    fii_dii_path = Path("data/raw/fii_dii/fii_dii_cache.parquet")
    
    if not fii_dii_path.exists():
        print(f"❌ File not found: {fii_dii_path}")
        return False
    
    df = pd.read_parquet(fii_dii_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df.head(10).to_string())
    
    if 'date' in df.columns:
        print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    
    # Check if it's real data
    if len(df) < 100:
        print("⚠️ WARNING: Very few rows - may not have enough historical data")
    
    return True


def main():
    delivery_ok = inspect_delivery_data()
    fii_dii_ok = inspect_fii_dii_data()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Delivery Data: {'✅ OK' if delivery_ok else '❌ NEEDS BACKFILL'}")
    print(f"FII/DII Data: {'✅ OK' if fii_dii_ok else '❌ NEEDS BACKFILL'}")
    
    if not delivery_ok:
        print("\n⚠️ ACTION REQUIRED: Run fetch_historical_delivery_data.py to backfill real NSE data")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
