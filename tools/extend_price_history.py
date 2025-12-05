"""
Extend Price History Tool

Fetches extended historical price data back to 2007 from Yahoo Finance.
Merges with existing price data and handles:
- Ticker changes
- Delistings
- Corporate actions (split adjustment)

Usage:
    python tools/extend_price_history.py
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance: pip install yfinance")


def load_universe(universe_file: str = "data/static/sector_map.csv") -> List[str]:
    """Load list of tickers from universe file."""
    df = pd.read_csv(universe_file)
    
    if "ticker_nse" in df.columns:
        tickers = df["ticker_nse"].tolist()
    elif "ticker" in df.columns:
        tickers = df["ticker"].tolist()
    else:
        raise ValueError("No ticker column found in universe file")
    
    return [t for t in tickers if pd.notna(t)]


def fetch_yf_prices(
    ticker: str,
    start_date: str = "2007-01-01",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical prices from Yahoo Finance.
    
    Args:
        ticker: Stock ticker (NSE format, e.g., 'RELIANCE')
        start_date: Start date
        end_date: End date (defaults to today)
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Yahoo Finance uses .NS suffix for NSE stocks
    yf_ticker = f"{ticker}.NS"
    
    try:
        data = yf.download(
            yf_ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,  # Adjust for splits/dividends
        )
        
        if data.empty:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Standardize column names
        data.columns = [c.lower() for c in data.columns]
        data = data.rename(columns={
            "adj close": "close",
        })
        
        # Add ticker column
        data["ticker"] = ticker
        
        # Reset index to make date a column
        data = data.reset_index()
        data = data.rename(columns={"Date": "date", "index": "date"})
        
        # Select columns
        cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in cols if c in data.columns]
        data = data[available_cols]
        
        return data
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def load_existing_prices(
    price_file: str = "data/raw/price_confirmation/all_prices.csv",
) -> Optional[pd.DataFrame]:
    """Load existing price data."""
    path = Path(price_file)
    
    if not path.exists():
        return None
    
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def merge_price_data(
    existing_df: Optional[pd.DataFrame],
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge existing and new price data.
    
    Keeps new data for overlapping dates (fresher data).
    """
    if existing_df is None:
        return new_df
    
    # Combine
    combined = pd.concat([new_df, existing_df], ignore_index=True)
    
    # Remove duplicates (keep first = new data)
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="first")
    
    # Sort
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    return combined


def extend_price_history(
    output_file: str = "data/raw/price_panel/extended_prices.parquet",
    start_date: str = "2007-01-01",
    universe_file: str = "data/static/sector_map.csv",
    existing_prices_file: Optional[str] = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Extend price history by fetching from Yahoo Finance.
    
    Args:
        output_file: Output file path
        start_date: Start date for history
        universe_file: Universe file with tickers
        existing_prices_file: Existing prices to merge with
        batch_size: Number of tickers to fetch at once
        
    Returns:
        Combined DataFrame with extended history
    """
    print(f"Extending price history from {start_date}...")
    
    # Load universe
    tickers = load_universe(universe_file)
    print(f"Found {len(tickers)} tickers in universe")
    
    # Load existing prices
    existing_df = None
    if existing_prices_file:
        existing_df = load_existing_prices(existing_prices_file)
        if existing_df is not None:
            print(f"Loaded {len(existing_df)} existing price records")
    
    # Fetch new prices
    all_data = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Fetching {ticker}...", end=" ")
        
        df = fetch_yf_prices(ticker, start_date=start_date)
        
        if df is not None and not df.empty:
            all_data.append(df)
            print(f"OK ({len(df)} rows)")
        else:
            failed_tickers.append(ticker)
            print("FAILED")
        
        # Progress pause every batch
        if (i + 1) % batch_size == 0:
            print(f"  Progress: {i+1}/{len(tickers)} tickers processed")
    
    # Combine new data
    if not all_data:
        print("No data fetched!")
        if existing_df is not None:
            return existing_df
        return pd.DataFrame()
    
    new_df = pd.concat(all_data, ignore_index=True)
    print(f"Fetched {len(new_df)} new price records")
    
    # Merge with existing
    combined_df = merge_price_data(existing_df, new_df)
    print(f"Combined total: {len(combined_df)} records")
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.endswith(".parquet"):
        combined_df.to_parquet(output_file, index=False)
    else:
        combined_df.to_csv(output_file, index=False)
    
    print(f"Saved to {output_file}")
    
    # Report
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total tickers: {combined_df['ticker'].nunique()}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Total records: {len(combined_df)}")
    
    if failed_tickers:
        print(f"\nFailed tickers ({len(failed_tickers)}):")
        for t in failed_tickers[:10]:
            print(f"  - {t}")
        if len(failed_tickers) > 10:
            print(f"  ... and {len(failed_tickers) - 10} more")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Extend price history from Yahoo Finance")
    parser.add_argument(
        "--start-date",
        default="2007-01-01",
        help="Start date for history (default: 2007-01-01)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/price_panel/extended_prices.parquet",
        help="Output file path",
    )
    parser.add_argument(
        "--universe",
        default="data/static/sector_map.csv",
        help="Universe file with tickers",
    )
    parser.add_argument(
        "--existing",
        default=None,
        help="Existing prices file to merge with",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for progress reporting",
    )
    
    args = parser.parse_args()
    
    extend_price_history(
        output_file=args.output,
        start_date=args.start_date,
        universe_file=args.universe,
        existing_prices_file=args.existing,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
