"""FII/DII Data Provider for NSE India.

Fetches institutional ownership flow data from NSE using NSEPython and pynse.
This data is crucial for predicting short-term stock movements based on
institutional buying/selling patterns.

Data sources:
- NSE official reports (FII/FPI and DII trading activity)
- Bulk/Block deals
- F&O participant-wise OI data
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class FIIDIIProvider:
    """Provider for FII/DII institutional flow data from NSE.

    Reads from the consolidated parquet cache built by
    scripts/fetch_fii_dii_data.py (uses requests directly — no nsepython).

    Cache location: data/raw/fii_dii/fii_dii_cache.parquet
    Columns:  date, fii_cash_buy, fii_cash_sell, fii_cash_net,
              dii_cash_buy, dii_cash_sell, dii_cash_net
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(settings.data_dir) / "raw" / "fii_dii"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._consolidated_cache = self.cache_dir / "fii_dii_cache.parquet"

    def fetch_fii_dii_data(
        self,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return FII/DII daily data for the requested date range.

        Reads the consolidated cache built by fetch_fii_dii_data.py.
        Returns empty DataFrame if cache doesn't exist (run the script first).
        """
        if not self._consolidated_cache.exists():
            logger.warning(
                "FII/DII cache not found. Run: python scripts/fetch_fii_dii_data.py"
            )
            return pd.DataFrame()

        df = pd.read_parquet(self._consolidated_cache)
        df["date"] = pd.to_datetime(df["date"])

        # Filter to requested range
        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
        subset = df[mask].copy()

        if subset.empty:
            logger.warning(
                f"No FII/DII data in cache for {start_date.date()} – {end_date.date()}. "
                "Run: python scripts/fetch_fii_dii_data.py"
            )
        else:
            logger.info(f"Loaded {len(subset)} FII/DII rows from cache")

        return subset
    
    def fetch_bulk_block_deals(
        self,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch bulk and block deals from NSE.
        
        These indicate large institutional/promoter transactions.
        """
        cache_path = self.cache_dir / f"bulk_block_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)
        
        if not self._nse_available:
            return pd.DataFrame()
        
        try:
            import nsepython as nse
            
            # Fetch bulk deals
            bulk_data = nse.block_deal()
            
            if bulk_data and isinstance(bulk_data, (list, dict)):
                if isinstance(bulk_data, dict):
                    bulk_data = [bulk_data]
                
                df = pd.DataFrame(bulk_data)
                df.to_parquet(cache_path, index=False)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching bulk/block deals: {e}")
            return pd.DataFrame()
    
    def _parse_value(self, value) -> float:
        """Parse numeric value from potentially string input."""
        if pd.isna(value) or value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove commas and currency symbols
            value = value.replace(',', '').replace('₹', '').replace('Rs', '').strip()
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0


class DeliveryDataProvider:
    """Provider for NSE delivery percentage data.
    
    Delivery % indicates conviction buying - high delivery %
    means traders are taking positions rather than day trading.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(settings.data_dir) / "raw" / "delivery"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_delivery_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch delivery percentage data for a symbol.
        
        Returns DataFrame with:
        - date
        - symbol
        - delivery_qty
        - traded_qty
        - delivery_pct
        """
        cache_path = self.cache_dir / f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)
        
        try:
            from nsepy import get_history
            
            # nsepy includes delivery data in historical data
            df = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date,
                series='EQ'
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Reset index to get date as column
            df = df.reset_index()
            
            # Extract relevant columns
            result = pd.DataFrame({
                'date': df['Date'],
                'symbol': symbol,
                'delivery_qty': df.get('Deliverable Volume', df.get('Deliverble Volume', 0)),
                'traded_qty': df.get('Total Traded Quantity', df.get('Volume', 0)),
                'delivery_pct': df.get('% Deliverble', df.get('%Deliverble', 0))
            })
            
            result.to_parquet(cache_path, index=False)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching delivery data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_bulk_delivery_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch delivery data for multiple symbols."""
        all_data = []
        
        for symbol in symbols:
            df = self.fetch_delivery_data(symbol, start_date, end_date, use_cache)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


def get_fii_dii_provider() -> FIIDIIProvider:
    """Factory function to get FII/DII provider."""
    return FIIDIIProvider()


def get_delivery_provider() -> DeliveryDataProvider:
    """Factory function to get delivery data provider."""
    return DeliveryDataProvider()
