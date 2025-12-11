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
    
    Fetches:
    - Daily FII/DII cash segment net buying
    - F&O segment activity
    - Sector-wise FII flows
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(settings.data_dir) / "raw" / "fii_dii"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._nse_available = self._check_nse_libs()
    
    def _check_nse_libs(self) -> bool:
        """Check if NSE libraries are available."""
        try:
            import nsepython
            return True
        except ImportError:
            try:
                from nsepy import get_history
                return True
            except ImportError:
                logger.warning("NSE libraries not available. Install with: pip install nsepython nsepy")
                return False
    
    def fetch_fii_dii_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch FII/DII daily trading data.
        
        Returns DataFrame with columns:
        - date: Trading date
        - fii_cash_buy: FII buy value (Rs Cr)
        - fii_cash_sell: FII sell value (Rs Cr)
        - fii_cash_net: FII net buy/sell (Rs Cr)
        - dii_cash_buy: DII buy value (Rs Cr)
        - dii_cash_sell: DII sell value (Rs Cr)
        - dii_cash_net: DII net buy/sell (Rs Cr)
        - fii_index_futures_long: FII long positions in index futures
        - fii_index_futures_short: FII short positions in index futures
        """
        cache_path = self.cache_dir / f"fii_dii_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading FII/DII data from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        
        if not self._nse_available:
            logger.warning("NSE libraries not available, returning empty DataFrame")
            return pd.DataFrame()
        
        try:
            import nsepython as nse
            
            # Fetch FII/DII data using nsepython
            data_list = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    date_str = current_date.strftime("%d-%b-%Y")
                    
                    # Get FII/DII data for the date
                    fii_data = nse.fii_dii(date_str)
                    
                    if fii_data and isinstance(fii_data, (list, dict)):
                        if isinstance(fii_data, dict):
                            fii_data = [fii_data]
                        
                        for record in fii_data:
                            data_list.append({
                                'date': current_date,
                                'fii_cash_buy': self._parse_value(record.get('FII/FPIs_buyValue', 0)),
                                'fii_cash_sell': self._parse_value(record.get('FII/FPIs_sellValue', 0)),
                                'fii_cash_net': self._parse_value(record.get('FII/FPIs_netValue', 0)),
                                'dii_cash_buy': self._parse_value(record.get('DII_buyValue', 0)),
                                'dii_cash_sell': self._parse_value(record.get('DII_sellValue', 0)),
                                'dii_cash_net': self._parse_value(record.get('DII_netValue', 0)),
                            })
                
                except Exception as e:
                    logger.debug(f"No FII/DII data for {current_date}: {e}")
                
                current_date += timedelta(days=1)
            
            if data_list:
                df = pd.DataFrame(data_list)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').drop_duplicates(subset=['date'])
                
                # Save to cache
                df.to_parquet(cache_path, index=False)
                logger.info(f"Saved FII/DII data to cache: {cache_path}")
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching FII/DII data: {e}")
            return pd.DataFrame()
    
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
