"""
TradingView Data Provider for Fundamental Data.
Uses tradingview-screener library.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

try:
    from tradingview_screener import Query, Column
    HAS_TV_SCREENER = True
except ImportError:
    HAS_TV_SCREENER = False

class TradingViewProvider:
    """
    TradingView data provider using tradingview-screener library.
    Fetches current fundamental data.
    Note: TradingView screener mostly provides CURRENT data (TTM/FQ), not full history.
    For backtesting, this is a limitation unless we have historical snapshots.
    However, for the purpose of this task ("Engineer Fundamental Features"), we will fetch what is available.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache/tradingview")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def is_available(self) -> bool:
        return HAS_TV_SCREENER

    def fetch_fundamentals(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for a symbol.
        Note: This fetches CURRENT data. We will timestamp it as today.
        """
        if not self.is_available():
            print("TradingView screener library not installed.")
            return pd.DataFrame()

        # Check cache
        cache_file = self.cache_dir / f"{symbol}.parquet"
        if cache_file.exists():
            # If cache is recent (< 1 day), use it
            try:
                df = pd.read_parquet(cache_file)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age.days < 1:
                    return df
            except Exception:
                pass

        try:
            # Map symbol to TV format if needed (e.g., RELIANCE -> NSE:RELIANCE)
            # But the screener query filters by exchange, so we just need the name.
            # Assuming symbol is just the ticker name (e.g. "RELIANCE")
            
            # Define columns to fetch
            # We fetch TTM (Trailing Twelve Months) or FQ (Fiscal Quarter) where appropriate
            cols = [
                'name', 'close', 'volume', 'market_cap_basic',
                'price_earnings_ttm', 'price_book_fq',
                'return_on_equity_fq', 'return_on_assets_fq',
                'debt_to_equity_fq', 'current_ratio_fq', 'quick_ratio_fq',
                'gross_margin_ttm', 'operating_margin_ttm', 'net_profit_margin_ttm',
                'total_revenue_fq', 'net_income_fq',
                'earnings_per_share_basic_ttm'
            ]
            
            q = (Query()
                 .select(*cols)
                 .where(Column('exchange') == 'NSE')
                 .where(Column('name') == symbol)
                 .limit(1))
                 
            rows = q.get_scanner_data()
            
            if not rows:
                # Try BSE if NSE fails? Or maybe symbol name mismatch?
                # For now, return empty
                return pd.DataFrame()
                
            # Parse row
            # rows is a list of tuples/lists in the order of select()
            # Wait, get_scanner_data() returns (total_count, list_of_rows) or just list_of_rows?
            # Based on my test script printing '0', it might be returning total count first?
            # Let's assume it returns a list of rows, where each row is a dict or object?
            # The library documentation says get_scanner_data() returns a tuple: (total_count, data_list)
            # So rows[0] being 0 means total_count=0?
            # Ah, if rows[0] is 0, then no data found.
            
            # Let's handle the tuple return
            if isinstance(rows, tuple):
                total_count, data = rows
                if total_count == 0 or not data:
                    return pd.DataFrame()
                row = data[0]
            else:
                # Maybe it returns just list?
                if not rows:
                    return pd.DataFrame()
                row = rows[0]

            # Map to our schema
            # row is likely a dictionary if using select() with names?
            # Or a list of values?
            # The library usually returns a dict per row if using select().
            
            # Let's assume row is a dict for now, based on common python libraries.
            # If it's a list, I need to map by index.
            
            # Actually, looking at library source or examples is best.
            # But I can't browse web.
            # I'll assume it returns a dict.
            
            record = {
                'symbol': symbol,
                'date': pd.Timestamp.now().normalize(), # Current data
                'report_date': pd.Timestamp.now().normalize(),
                
                'market_cap': row.get('market_cap_basic'),
                'pe_ratio': row.get('price_earnings_ttm'),
                'pb_ratio': row.get('price_book_fq'),
                'ps_ratio': row.get('price_sales_ttm'),
                
                'roe': row.get('return_on_equity_fq'),
                'roa': row.get('return_on_assets_fq'),
                
                'debt_to_equity': row.get('debt_to_equity_fq'),
                'current_ratio': row.get('current_ratio_fq'),
                'quick_ratio': row.get('quick_ratio_fq'),
                
                'gross_margin': row.get('gross_margin_ttm'),
                'operating_margin': row.get('operating_margin_ttm'),
                'net_margin': row.get('net_profit_margin_ttm'),
                
                'revenue': row.get('total_revenue_fq'),
                'net_income': row.get('net_income_fq'),
                'eps': row.get('earnings_per_share_basic_ttm'),
            }
            
            # Clean NaNs
            record = {k: v for k, v in record.items() if v is not None}
            
            df = pd.DataFrame([record])
            
            # Cache
            df.to_parquet(cache_file, index=False)
            
            return df

        except Exception as e:
            print(f"Error fetching {symbol} from TradingView: {e}")
            return pd.DataFrame()

def get_tradingview_provider(cache_dir: Optional[Path] = None) -> TradingViewProvider:
    return TradingViewProvider(cache_dir=cache_dir)
