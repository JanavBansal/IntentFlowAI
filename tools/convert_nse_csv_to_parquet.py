#!/usr/bin/env python
"""Convert NSE scraper CSV files to single parquet format.

This script processes all CSV files downloaded by the nse_scrap tool,
standardizes column names, adds sector information, and creates a
consolidated parquet file for the training pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def load_sector_mapping(sector_map_path: Path) -> dict:
    """Load sector mapping from CSV."""
    if not sector_map_path.exists():
        logger.warning(f"Sector map not found at {sector_map_path}")
        return {}
    
    try:
        df = pd.read_csv(sector_map_path)
        return dict(zip(df['ticker'], df['sector']))
    except Exception as e:
        logger.warning(f"Failed to load sector map: {e}")
        return {}


    def standardize_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize DataFrame columns to match expected schema."""
        # NSE CSVs have trailing spaces in column names - strip them
        df.columns = df.columns.str.strip()
        
        # Standard column mapping (NSE format)
        column_map = {
            'Date': 'date',
            'Symbol': 'ticker',
            'Series': 'series',
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close',
            'Last Price': 'last',
            'Prev Close': 'prev_close',
            'Total Traded Quantity': 'volume',
            'Turnover ₹': 'traded_value',
            'No. of Trades': 'total_trades',
            'Deliverable Qty': 'delivery_qty',
            '% Dly Qt to Traded Qty': 'delivery_pct',
            'Average Price': 'vwap',
        }
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Ensure ticker column is present
        if 'ticker' not in df.columns and symbol:
            df['ticker'] = symbol
        
        # Parse date (NSE format: "01-Jan-2020")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        
        # Clean numeric columns (remove commas)
        numeric_cols = ['open', 'high', 'low', 'close', 'last', 'prev_close', 
                       'volume', 'traded_value', 'total_trades', 'delivery_qty', 'delivery_pct', 'vwap']
        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and convert to float
                df[col] = df[col].astype(str).str.replace(',', '').replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


def convert_nse_data(
    csv_dir: Path,
    sector_map_path: Path,
    output_path: Path,
    min_rows_per_file: int = 10
) -> pd.DataFrame:
    """Convert all NSE CSV files to consolidated parquet.
    
    Args:
        csv_dir: Directory containing stock folders with CSV files
        sector_map_path: Path to sector mapping CSV
        output_path: Output parquet file path
        min_rows_per_file: Minimum rows to consider a file valid
    
    Returns:
        Combined DataFrame
    """
    logger.info(f"Loading sector mapping from {sector_map_path}")
    sector_map = load_sector_mapping(sector_map_path)
    logger.info(f"Loaded {len(sector_map)} sector mappings")
    
    all_frames = []
    processed = 0
    failed = 0
    
    # Find all CSV files
    csv_files = list(csv_dir.glob("*/*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        df = standardize_nse_csv(csv_file, sector_map)
        
        if not df.empty and len(df) >= min_rows_per_file:
            all_frames.append(df)
            processed += 1
            
            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(csv_files)} files...")
        else:
            failed += 1
    
    if not all_frames:
        logger.error("No valid data found!")
        return pd.DataFrame()
    
    # Combine all frames
    logger.info("Combining all dataframes...")
    combined = pd.concat(all_frames, ignore_index=True)
    
    # Remove duplicates
    combined = combined.drop_duplicates(subset=['date', 'ticker'])
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Fill missing delivery data with 0 (for older data that might not have it)
    if 'delivery_qty' in combined.columns:
        combined['delivery_qty'] = combined['delivery_qty'].fillna(0)
    if 'delivery_pct' in combined.columns:
        combined['delivery_pct'] = combined['delivery_pct'].fillna(0)
    
    logger.info(
        f"Processed {processed} files successfully, {failed} failed",
        extra={
            "total_rows": len(combined),
            "tickers": combined['ticker'].nunique(),
            "date_range": f"{combined['date'].min().date()} to {combined['date'].max().date()}",
        },
    )
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    return combined


def verify_data_quality(df: pd.DataFrame, min_trading_days: int = 250):
    """Print data quality report."""
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Trading days: {df['date'].nunique()}")
    
    print(f"\n📈 Ticker Coverage:")
    ticker_counts = df.groupby('ticker').size().sort_values(ascending=False)
    tickers_above_threshold = (ticker_counts >= min_trading_days).sum()
    print(f"  Tickers with {min_trading_days}+ days: {tickers_above_threshold}")
    print(f"  Top 10 by data points:")
    for ticker, count in ticker_counts.head(10).items():
        print(f"    {ticker:15s}: {count:4d} days")
    
    print(f"\n🏢 Sector Distribution:")
    if 'sector' in df.columns:
        sector_counts = df['sector'].value_counts()
        for sector, count in sector_counts.items():
            pct = 100 * count / len(df)
            print(f"    {sector:25s}: {count:6,} rows ({pct:5.1f}%)")
    
    print(f"\n📦 Delivery Data:")
    if 'delivery_qty' in df.columns and 'delivery_pct' in df.columns:
        has_delivery = (df['delivery_qty'] > 0).sum()
        pct = 100 * has_delivery / len(df)
        print(f"  Rows with delivery data: {has_delivery:,} ({pct:.1f}%)")
        print(f"  Avg delivery %: {df[df['delivery_pct'] > 0]['delivery_pct'].mean():.2f}%")
    else:
        print("  ❌ Delivery columns not found")
    
    print(f"\n✅ Missing Values:")
    missing = df.isnull().sum()
    if missing.any():
        for col, count in missing[missing > 0].items():
            pct = 100 * count / len(df)
            print(f"    {col:15s}: {count:6,} ({pct:5.1f}%)")
    else:
        print("  No missing values!")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert NSE CSV files to parquet")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=ROOT / "data" / "external" / "nse_scrap" / "data",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--sector-map",
        type=Path,
        default=ROOT / "data" / "static" / "sector_map.csv",
        help="Sector mapping CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "raw" / "price_confirmation" / "data.parquet",
        help="Output parquet file"
    )
    
    args = parser.parse_args()
    
    # Convert
    result = convert_nse_data(
        csv_dir=args.csv_dir,
        sector_map_path=args.sector_map,
        output_path=args.output
    )
    
    if not result.empty:
        verify_data_quality(result, min_trading_days=settings.min_trading_days)
        print(f"✅ Success! Data saved to: {args.output}")
    else:
        print("❌ Failed to convert data")
        sys.exit(1)
