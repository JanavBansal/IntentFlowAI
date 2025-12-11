
import pandas as pd
from pathlib import Path

def inspect_data():
    base_dir = Path("data")
    
    # 1. Inspect Price Data
    price_path = base_dir / "processed" / "prices.parquet"
    if not price_path.exists():
        price_path = base_dir / "raw" / "price_confirmation" / "all_prices.csv"
        print(f"Checking Price Data at: {price_path}")
        df_price = pd.read_csv(price_path)
    else:
        print(f"Checking Price Data at: {price_path}")
        df_price = pd.read_parquet(price_path)
    
    if "date" in df_price.columns:
        df_price["date"] = pd.to_datetime(df_price["date"])
        print(f"Price Date Range: {df_price['date'].min()} to {df_price['date'].max()}")
    
    print(f"Price Columns: {df_price.columns.tolist()}")
    print(f"Total Price Rows: {len(df_price)}")
    print(f"Unique Tickers: {df_price['ticker'].nunique()}")
    
    if "sector" in df_price.columns:
        print(f"Sectors Present: Yes ({df_price['sector'].nunique()} unique)")
        print(f"Missing Sectors: {df_price['sector'].isna().sum()}")
    else:
        print("Sectors Present: No (Check sector_map.csv)")

    # 2. Inspect Fundamental Data
    fund_path = base_dir / "processed" / "fundamentals_eodhd.parquet"
    if not fund_path.exists():
        print("Primary Fundamental Data (EODHD) NOT FOUND.")
        fund_path = base_dir / "processed" / "fundamentals_new_batch.parquet"
    
    if fund_path.exists():
        print(f"\nChecking Fundamental Data at: {fund_path}")
        df_fund = pd.read_parquet(fund_path)
        if "date" in df_fund.columns:
             df_fund["date"] = pd.to_datetime(df_fund["date"])
             print(f"Fundamental Date Range: {df_fund['date'].min()} to {df_fund['date'].max()}")
        elif "report_date" in df_fund.columns:
             df_fund["report_date"] = pd.to_datetime(df_fund["report_date"])
             print(f"Fundamental Date Range (Report): {df_fund['report_date'].min()} to {df_fund['report_date'].max()}")

        print(f"Fundamental Columns (Sample): {df_fund.columns.tolist()[:10]}...")
        print(f"Total Fundamental Rows: {len(df_fund)}")
        print(f"Unique Fundamental Tickers: {df_fund['ticker'].nunique() if 'ticker' in df_fund.columns else 'Unknown'}")
    else:
        print("\nNo Fundamental Data Found in processed/")

    # 3. Check Sector Map
    sector_path = base_dir / "static" / "sector_map.csv"
    if sector_path.exists():
        print(f"\nChecking Sector Map at: {sector_path}")
        df_sector = pd.read_csv(sector_path)
        print(f"Total Tickers in Sector Map: {len(df_sector)}")
        print(f"Sector Map Columns: {df_sector.columns.tolist()}")

if __name__ == "__main__":
    inspect_data()
