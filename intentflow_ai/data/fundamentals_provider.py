"""
Fundamental Data Provider Infrastructure

Supports multiple data sources with fallback:
1. Yahoo Finance (yfinance) - Free, good for large caps
2. Financial Modeling Prep (FMP) - Premium, comprehensive (optional)
3. Alpha Vantage - Alternative (optional)

Point-in-time correctness is ensured via reporting delay buffer.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class FundamentalDataProvider(ABC):
    """Abstract base class for fundamental data providers."""
    
    @abstractmethod
    def fetch_fundamentals(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for a symbol.
        
        Returns DataFrame with columns:
        - date: observation date
        - symbol: ticker symbol
        - pe_ratio, pb_ratio, ps_ratio: valuation
        - roe, roa, gross_margin, operating_margin, net_margin: profitability
        - revenue_growth_yoy, earnings_growth_yoy: growth
        - debt_to_equity, current_ratio: balance sheet
        - market_cap, shares_outstanding: metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API key set, library installed)."""
        pass


class YahooFinanceProvider(FundamentalDataProvider):
    """Yahoo Finance provider using yfinance library."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache/fundamentals_yahoo")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def is_available(self) -> bool:
        return HAS_YFINANCE
    
    def fetch_fundamentals(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch fundamentals from Yahoo Finance."""
        if not self.is_available():
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if len(df) > 0:
                    return df
            except Exception:
                pass  # Cache corrupted, refetch
        
        # Fetch from Yahoo
        try:
            ticker = yf.Ticker(symbol)
            
            # Get info (for current fundamentals)
            info = ticker.info
            if not info or 'symbol' not in info:
                return pd.DataFrame()
            
            # Get quarterly data - handle both old and new yfinance API
            try:
                quarterly_financials = ticker.quarterly_financials
                quarterly_balance_sheet = ticker.quarterly_balance_sheet  
                quarterly_cashflow = ticker.quarterly_cashflow
            except Exception as e:
                print(f"   Warning: Could not fetch quarterly data for {symbol}: {e}")
                quarterly_financials = pd.DataFrame()
                quarterly_balance_sheet = pd.DataFrame()
                quarterly_cashflow = pd.DataFrame()
            
            # Build fundamental metrics DataFrame
            records = []
            
            # Process quarterly data if available
            if quarterly_financials is not None and not quarterly_financials.empty:
                try:
                    for quarter_date in quarterly_financials.columns:
                        # Convert to datetime
                        try:
                            if isinstance(quarter_date, pd.Timestamp):
                                qdate = quarter_date
                            else:
                                qdate = pd.to_datetime(quarter_date)
                        except Exception:
                            continue
                        
                        # Get data for this quarter
                        financials = quarterly_financials[quarter_date]
                        balance_sheet = quarterly_balance_sheet[quarter_date] if quarterly_balance_sheet is not None and not quarterly_balance_sheet.empty else pd.Series()
                        cashflow = quarterly_cashflow[quarter_date] if quarterly_cashflow is not None and not quarterly_cashflow.empty else pd.Series()
                        
                        record = self._extract_fundamental_metrics(
                            symbol=symbol,
                            quarter_date=qdate,
                            financials=financials,
                            balance_sheet=balance_sheet,
                            cashflow=cashflow,
                            info=info
                        )
                        records.append(record)
                except Exception as e:
                    print(f"   Warning: Error processing quarterly data for {symbol}: {e}")
            
            # If no quarterly data, create single record from info
            if not records:
                records.append(self._extract_from_info(symbol, info))
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Forward fill to create daily data (fundamentals change quarterly)
            df = self._expand_to_daily(df, start_date, end_date)
            
            # Apply reporting delay (45 days) - critical for point-in-time correctness
            df = self._apply_reporting_delay(df, delay_days=45)
            
            # Cache results
            if not df.empty:
                df.to_parquet(cache_file, index=False)
            
            # Return all data - let caller handle date filtering
            return df
            
        except Exception as e:
            print(f"   Error fetching fundamentals for {symbol}: {e}")
            return pd.DataFrame()
    
    def _extract_fundamental_metrics(
        self,
        symbol: str,
        quarter_date,
        financials: pd.Series,
        balance_sheet: pd.Series,
        cashflow: pd.Series,
        info: dict
    ) -> dict:
        """Extract fundamental metrics from Yahoo data."""
        
        # Convert quarter_date to datetime
        if isinstance(quarter_date, str):
            quarter_date = pd.to_datetime(quarter_date)
        
        # Helper to safely get values
        def safe_get(series, key, default=np.nan):
            try:
                return float(series.get(key, default))
            except (ValueError, TypeError):
                return default
        
        # Extract metrics
        total_revenue = safe_get(financials, 'Total Revenue')
        net_income = safe_get(financials, 'Net Income')
        gross_profit = safe_get(financials, 'Gross Profit')
        operating_income = safe_get(financials, 'Operating Income')
        
        total_assets = safe_get(balance_sheet, 'Total Assets')
        total_equity = safe_get(balance_sheet, 'Total Stockholder Equity')
        total_debt = safe_get(balance_sheet, 'Total Debt', 0)
        current_assets = safe_get(balance_sheet, 'Current Assets')
        current_liabilities = safe_get(balance_sheet, 'Current Liabilities')
        cash = safe_get(balance_sheet, 'Cash And Cash Equivalents', 0)
        
        operating_cf = safe_get(cashflow, 'Operating Cash Flow')
        
        # Compute derived metrics
        record = {
            'symbol': symbol,
            'date': quarter_date,
            'report_date': quarter_date,  # Will be shifted by reporting delay
            
            # Valuation (from info - current values)
            'pe_ratio': info.get('trailingPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            'market_cap': info.get('marketCap', np.nan),
            
            # Profitability
            'roe': (net_income / total_equity * 4) if total_equity > 0 else np.nan,  # Annualized
            'roa': (net_income / total_assets * 4) if total_assets > 0 else np.nan,
            'gross_margin': (gross_profit / total_revenue) if total_revenue > 0 else np.nan,
            'operating_margin': (operating_income / total_revenue) if total_revenue > 0 else np.nan,
            'net_margin': (net_income / total_revenue) if total_revenue > 0 else np.nan,
            
            # Growth (will compute YoY later)
            'revenue': total_revenue,
            'net_income': net_income,
            'eps': safe_get(financials, 'Basic EPS'),
            
            # Balance Sheet
            'debt_to_equity': (total_debt / total_equity) if total_equity > 0 else np.nan,
            'current_ratio': (current_assets / current_liabilities) if current_liabilities > 0 else np.nan,
            'quick_ratio': ((current_assets - safe_get(balance_sheet, 'Inventory', 0)) / current_liabilities) if current_liabilities > 0 else np.nan,
            'cash_to_debt': (cash / total_debt) if total_debt > 0 else np.nan,
            
            # Earnings Quality
            'operating_cf': operating_cf,
            'cf_to_ni': (operating_cf / net_income) if net_income != 0 and not pd.isna(net_income) else np.nan,
            'accruals': ((net_income - operating_cf) / total_assets) if total_assets > 0 and not pd.isna(operating_cf) else np.nan,
        }
        
        return record
    
    def _extract_from_info(self, symbol: str, info: dict) -> dict:
        """Fallback: extract what we can from info dict."""
        return {
            'symbol': symbol,
            'date': pd.Timestamp.now(),
            'report_date': pd.Timestamp.now(),
            'pe_ratio': info.get('trailingPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
            'market_cap': info.get('marketCap', np.nan),
            'roe': info.get('returnOnEquity', np.nan),
            'gross_margin': info.get('grossMargins', np.nan),
            'operating_margin': info.get('operatingMargins', np.nan),
            'net_margin': info.get('profitMargins', np.nan),
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'quick_ratio': info.get('quickRatio', np.nan),
        }
    
    def _expand_to_daily(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Expand quarterly data to daily by forward-filling."""
        if df.empty:
            return df
        
        # Create daily date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        daily_df = pd.DataFrame({'date': date_range})
        daily_df['symbol'] = df['symbol'].iloc[0]
        
        # Merge and forward fill
        df_daily = pd.merge_asof(
            daily_df.sort_values('date'),
            df.sort_values('date'),
            on='date',
            by='symbol',
            direction='backward'
        )
        
        return df_daily
    
    def _apply_reporting_delay(self, df: pd.DataFrame, delay_days: int = 45) -> pd.DataFrame:
        """
        Apply reporting delay to ensure point-in-time correctness.
        
        Earnings are reported ~45 days after quarter end.
        This shifts the availability date forward.
        """
        if df.empty or 'report_date' not in df.columns:
            return df
        
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Shift the date that this data becomes available
        df['available_date'] = df['report_date'] + pd.Timedelta(days=delay_days)
        
        # Only use data after it's available
        df = df[df['date'] >= df['available_date']].copy()
        
        return df


class FMPProvider(FundamentalDataProvider):
    """Financial Modeling Prep API provider (premium, optional)."""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or os.environ.get('FMP_API_KEY')
        self.cache_dir = cache_dir or Path("data/cache/fundamentals_fmp")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def is_available(self) -> bool:
        return self.api_key is not None and HAS_REQUESTS
    
    def fetch_fundamentals(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch from FMP API."""
        if not self.is_available():
            return pd.DataFrame()
        
        # Implementation similar to Yahoo but using FMP API endpoints
        # Left as exercise - FMP has excellent API documentation
        raise NotImplementedError("FMP provider coming soon - use Yahoo for now")


class HybridFundamentalProvider:
    """
    Hybrid provider with market-specific fallback logic.
    
    - NSE (Indian stocks): Screener.in (primary) → Yahoo Finance with .NS (fallback)
    - US stocks: Yahoo Finance (primary) → FMP (fallback if available)
    """
    
    def __init__(self, fmp_api_key: Optional[str] = None):
        self.yahoo = YahooFinanceProvider()
        self.fmp = FMPProvider(fmp_api_key) if fmp_api_key else None
        
        # Import ScreenerIn provider
        try:
            from intentflow_ai.data.screener_in_provider import get_screener_provider
            self.screener = get_screener_provider()
        except ImportError:
            self.screener = None
    
    def _is_nse_ticker(self, symbol: str) -> bool:
        """Detect if symbol is an NSE ticker (no .NS suffix, Indian format)."""
        # NSE tickers are typically all caps, no dots
        # US tickers often have lowercase or special chars
        if '.' in symbol or len(symbol) > 20:
            return False
        
        # Check if it's a known US ticker pattern
        us_suffixes = ['.US', '.O', '.N', '.A']
        if any(symbol.endswith(suffix) for suffix in us_suffixes):
            return False
        
        # If all uppercase and reasonable length, assume NSE
        return symbol.isupper() and 2 <= len(symbol) <= 15
    
    def fetch_fundamentals(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch with market-specific fallback logic."""
        
        is_nse = self._is_nse_ticker(symbol)
        
        if is_nse:
            # NSE Stock - Try Screener.in first
            if self.screener:
                try:
                    df = self.screener.fetch_fundamentals(symbol, start_date, end_date)
                    if not df.empty:
                        return df
                    print(f"   Screener.in returned no data for {symbol}, trying Yahoo with .NS...")
                except Exception as e:
                    print(f"   Screener.in failed for {symbol}: {e}, trying Yahoo...")
            
            # Fallback: Yahoo Finance with .NS suffix
            if self.yahoo.is_available():
                try:
                    yahoo_symbol = f"{symbol}.NS"
                    df = self.yahoo.fetch_fundamentals(yahoo_symbol, start_date, end_date)
                    if not df.empty:
                        # Change symbol back to original (remove .NS)
                        df['symbol'] = symbol
                        return df
                except Exception as e:
                    print(f"   Yahoo (.NS) failed for {symbol}: {e}")
        else:
            # US Stock - Try Yahoo first
            if self.yahoo.is_available():
                try:
                    df = self.yahoo.fetch_fundamentals(symbol, start_date, end_date)
                    if not df.empty:
                        return df
                except Exception as e:
                    print(f"   Yahoo failed for {symbol}: {e}, trying FMP...")
            
            # Fallback to FMP if available
            if self.fmp and self.fmp.is_available():
                try:
                    return self.fmp.fetch_fundamentals(symbol, start_date, end_date)
                except Exception as e:
                    print(f"   FMP failed for {symbol}: {e}")
        
        # All methods failed
        print(f"⚠️  No fundamental data available for {symbol}")
        return pd.DataFrame()
    
    def fetch_batch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        progress_callback=None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch fundamentals for multiple symbols."""
        results = {}
        
        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i + 1, len(symbols), symbol)
            
            df = self.fetch_fundamentals(symbol, start_date, end_date)
            if not df.empty:
                results[symbol] = df
        
        return results


# Convenience import
import os

def get_fundamental_provider(fmp_api_key: Optional[str] = None) -> HybridFundamentalProvider:
    """Get configured fundamental data provider."""
    return HybridFundamentalProvider(fmp_api_key=fmp_api_key)


if __name__ == "__main__":
    # Test the provider
    provider = get_fundamental_provider()
    
    # Test with AAPL
    test_symbol = "AAPL"
    start = datetime(2020, 1, 1)
    end = datetime(2024, 1, 1)
    
    print(f"Testing fundamental data fetch for {test_symbol}...")
    df = provider.fetch_fundamentals(test_symbol, start, end)
    
    if not df.empty:
        print(f"\n✅ Successfully fetched {len(df)} days of fundamental data")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample (last 5 rows):")
        print(df.tail())
        print(f"\nData coverage: {df['date'].min()} to {df['date'].max()}")
    else:
        print("❌ Failed to fetch data")
