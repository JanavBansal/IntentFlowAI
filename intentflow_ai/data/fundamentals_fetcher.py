"""
Fundamental Data Fetcher - Batch download and caching

Fetches fundamental data for all universe symbols and caches results.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from intentflow_ai.config.settings import settings
from intentflow_ai.data.fundamentals_provider import get_fundamental_provider
from intentflow_ai.data.universe import load_universe


class FundamentalDataFetcher:
    """Batch fetch and cache fundamental data for universe."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(settings.data_dir) / "cache" / "fundamentals"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = get_fundamental_provider()
    
    def fetch_universe_fundamentals(
        self,
        universe_symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch fundamentals for all universe symbols.
        
        Args:
            universe_symbols: List of symbols (default: load from universe)
            start_date: Start date (default: 2005-01-01)
            end_date: End date (default: today)
            force_refresh: If True, ignore cache and refetch
            
        Returns:
            DataFrame with fundamentals for all symbols
        """
        # Load universe if not provided
        if universe_symbols is None:
            universe_df = load_universe()
            universe_symbols = universe_df['symbol'].unique().tolist()
        
        # Set default dates
        if start_date is None:
            start_date = datetime(2005, 1, 1)
        if end_date is None:
            end_date = datetime.now()
        
        # Check for cached consolidated file
        cache_file = self.cache_dir / "fundamentals_all.parquet"
        if not force_refresh and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter to requested range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # Check if we have all symbols
                cached_symbols = set(df['symbol'].unique())
                if set(universe_symbols).issubset(cached_symbols):
                    print(f"✅ Loaded {len(df)} rows of fundamentals from cache ({len(cached_symbols)} symbols)")
                    return df[df['symbol'].isin(universe_symbols)]
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}, re-fetching...")
        
        # Fetch data
        print(f"Fetching fundamentals for {len(universe_symbols)} symbols from {start_date.date()} to {end_date.date()}...")
        
        all_data = []
        failed_symbols = []
        
        for symbol in tqdm(universe_symbols, desc="Fetching fundamentals"):
            try:
                df = self.provider.fetch_fundamentals(symbol, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                print(f"   ⚠️  Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if not all_data:
            print("❌ No fundamental data fetched!")
            return pd.DataFrame()
        
        # Concatenate all data
        fundamentals_df = pd.concat(all_data, ignore_index=True)
        
        # Save consolidated cache
        fundamentals_df.to_parquet(cache_file, index=False)
        
        print(f"\n✅ Fetched fundamentals for {len(fundamentals_df['symbol'].unique())} symbols")
        print(f"   Total rows: {len(fundamentals_df):,}")
        print(f"   Date range: {fundamentals_df['date'].min().date()} to {fundamentals_df['date'].max().date()}")
        
        if failed_symbols:
            print(f"\n⚠️  Failed to fetch {len(failed_symbols)} symbols:")
            print(f"   {', '.join(failed_symbols[:10])}" + ("..." if len(failed_symbols) > 10 else ""))
        
        return fundamentals_df
    
    def get_fundamental_coverage(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze fundamental data coverage by symbol."""
        coverage = []
        
        for symbol in fundamentals_df['symbol'].unique():
            symbol_df = fundamentals_df[fundamentals_df['symbol'] == symbol]
            
            # Count non-null values for key metrics
            pe_coverage = symbol_df['pe_ratio'].notna().sum() / len(symbol_df)
            roe_coverage = symbol_df['roe'].notna().sum() / len(symbol_df)
            revenue_coverage = symbol_df['revenue'].notna().sum() / len(symbol_df) if 'revenue' in symbol_df.columns else 0
            
            coverage.append({
                'symbol': symbol,
                'total_days': len(symbol_df),
                'pe_coverage': pe_coverage,
                'roe_coverage': roe_coverage,
                'revenue_coverage': revenue_coverage,
                'avg_coverage': (pe_coverage + roe_coverage + revenue_coverage) / 3
            })
        
        coverage_df = pd.DataFrame(coverage).sort_values('avg_coverage', ascending=False)
        return coverage_df


if __name__ == "__main__":
    # Test fetcher
    fetcher = FundamentalDataFetcher()
    
    # Fetch for a small test set
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start = datetime(2020, 1, 1)
    end = datetime(2024, 1, 1)
    
    print("Testing fundamental data fetcher...")
    df = fetcher.fetch_universe_fundamentals(
        universe_symbols=test_symbols,
        start_date=start,
        end_date=end
    )
    
    if not df.empty:
        print("\n✅ Test successful!")
        print(f"\nSample data:")
        print(df.head())
        
        # Coverage analysis
        coverage = fetcher.get_fundamental_coverage(df)
        print(f"\nCoverage Analysis:")
        print(coverage)
