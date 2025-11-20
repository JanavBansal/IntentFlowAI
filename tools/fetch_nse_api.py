#!/usr/bin/env python
"""Fetch NSE stock data using official NSE API endpoint.

This module provides a direct API interface to NSE historical data,
based on the approach described in:
https://dev.to/singhanuj620/building-a-lightning-fast-nse-stock-data-scraper-from-api-calls-to-full-stack-web-app-4hl7
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class NSEAPIClient:
    """Client for NSE official API to fetch historical stock data."""
    
    BASE_URL = "https://www.nseindia.com"
    
    # Required headers to mimic browser and get session cookies
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/report-detail/eq_security',
        'Connection': 'keep-alive',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.cookies_initialized = False
    
    def initialize_session(self):
        """Initialize session by fetching cookies from NSE website."""
        try:
            # Visit the main page to get cookies
            response = self.session.get(
                f"{self.BASE_URL}/report-detail/eq_security",
                timeout=10
            )
            response.raise_for_status()
            
            self.cookies_initialized = True
            logger.info("NSE session initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NSE session: {e}")
            return False
    
    def fetch_historical_data(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        series: str = "EQ",
        retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            from_date: Start date in DD-MM-YYYY format
            to_date: End date in DD-MM-YYYY format
            series: Series type (default: "EQ" for equity)
            retries: Number of retry attempts
        
        Returns:
            DataFrame with historical data or None if failed
        """
        if not self.cookies_initialized:
            self.initialize_session()
        
        url = (
            f"{self.BASE_URL}/api/historical/generateSecurityWiseHistoricalData"
            f"?from={from_date}&to={to_date}&symbol={symbol}"
            f"&type=priceVolumeDeliverable&series={series}&csv=true"
        )
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                # Parse CSV data
                if response.text and len(response.text) > 10:
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    
                    # Standardize column names
                    df = self._standardize_columns(df, symbol)
                    return df
                else:
                    logger.warning(f"Empty response for {symbol}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed for {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch data for {symbol} after {retries} attempts")
                    return None
        
        return None
    
    def _standardize_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize DataFrame columns to match expected schema."""
        # Standard column mapping
        column_map = {
            'Date': 'date',
            'Symbol': 'ticker',
            'Series': 'series',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Last': 'last',
            'Prevclose': 'prev_close',
            'TOTTRDQTY': 'volume',
            'TOTTRDVAL': 'traded_value',
            'TIMESTAMP': 'timestamp',
            'TOTALTRADES': 'total_trades',
            'ISIN': 'isin',
            'DELIVERYQTY': 'delivery_qty',
            'DELIVERYPER': 'delivery_pct',
        }
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Ensure ticker column is present
        if 'ticker' not in df.columns and symbol:
            df['ticker'] = symbol
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'last', 'prev_close', 
                       'volume', 'traded_value', 'total_trades', 'delivery_qty', 'delivery_pct']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        from_date: str,
        to_date: str,
        max_workers: int = 5,
        delay_between_batches: float = 1.0
    ) -> pd.DataFrame:
        """Fetch data for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            from_date: Start date in DD-MM-YYYY format
            to_date: End date in DD-MM-YYYY format
            max_workers: Number of parallel workers
            delay_between_batches: Delay between batches to respect rate limits
        
        Returns:
            Combined DataFrame with all symbols
        """
        if not self.cookies_initialized:
            self.initialize_session()
        
        all_frames = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.fetch_historical_data, 
                    symbol, 
                    from_date, 
                    to_date
                ): symbol 
                for symbol in symbols
            }
            
            completed = 0
            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1
                
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_frames.append(df)
                        logger.info(f"[{completed}/{len(symbols)}] Fetched {len(df)} rows for {symbol}")
                    else:
                        logger.warning(f"[{completed}/{len(symbols)}] No data for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
                
                # Rate limiting
                if completed % 10 == 0:
                    time.sleep(delay_between_batches)
        
        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
            logger.info(f"Combined data: {len(combined)} rows, {combined['ticker'].nunique()} symbols")
            return combined
        else:
            logger.warning("No data fetched for any symbol")
            return pd.DataFrame()


def fetch_nse_data_by_year(
    symbols: list[str],
    start_year: int,
    end_year: Optional[int] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Fetch NSE data for multiple years.
    
    Args:
        symbols: List of stock symbols
        start_year: Starting year
        end_year: Ending year (default: current year)
        output_path: Optional path to save parquet file
    
    Returns:
        Combined DataFrame with all data
    """
    if end_year is None:
        end_year = datetime.now().year
    
    client = NSEAPIClient()
    all_frames = []
    
    for year in range(start_year, end_year + 1):
        from_date = f"01-01-{year}"
        to_date = f"31-12-{year}"
        
        logger.info(f"Fetching data for year {year}")
        df = client.fetch_multiple_symbols(symbols, from_date, to_date)
        
        if not df.empty:
            all_frames.append(df)
        
        # Rate limiting between years
        if year < end_year:
            time.sleep(2)
    
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['ticker', 'date'])
        combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(combined)} rows to {output_path}")
        
        return combined
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch NSE data using official API")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE", "TCS", "INFY", "HDFCBANK"],
                      help="Stock symbols to fetch")
    parser.add_argument("--start-year", type=int, default=2020,
                      help="Start year (default: 2020)")
    parser.add_argument("--end-year", type=int, default=None,
                      help="End year (default: current year)")
    parser.add_argument("--output", type=Path,
                      default=settings.data_dir / "raw" / "price_confirmation" / "data.parquet",
                      help="Output parquet file path")
    
    args = parser.parse_args()
    
    result = fetch_nse_data_by_year(
        symbols=args.symbols,
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output
    )
    
    if not result.empty:
        print(f"\n✅ Successfully fetched {len(result):,} rows")
        print(f"Date range: {result['date'].min().date()} to {result['date'].max().date()}")
        print(f"Symbols: {result['ticker'].nunique()}")
        print(f"Output: {args.output}")
    else:
        print("\n❌ No data fetched")
