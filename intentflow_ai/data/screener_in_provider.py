"""
Screener.in Data Provider for NSE Fundamental Data

Screener.in is the best free source for NSE fundamental data.
No official API, but clean HTML structure allows respectful web scraping.

Features extracted:
- Valuation: P/E, P/B, Market Cap, Dividend Yield
- Profitability: ROE, ROA, ROCE, Net Margin, OPM
- Growth: Sales growth, Profit growth  
- Leverage: Debt-to-Equity, Current Ratio
- Quality: Cash from operations, Return on Capital

Data structure: https://www.screener.in/company/{SYMBOL}/consolidated/
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


class ScreenerInProvider:
    """
    Screener.in data provider for NSE stocks.
    
    Uses web scraping with rate limiting and caching to be respectful.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, rate_limit_seconds: float = 2.0):
        """
        Initialize Screener.in provider.
        
        Args:
            cache_dir: Directory for caching responses
            rate_limit_seconds: Seconds to wait between requests (be respectful!)
        """
        self.base_url = "https://www.screener.in/company"
        self.cache_dir = cache_dir or Path("data/cache/screener_in")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit_seconds
        self.last_request_time = 0
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def fetch_fundamentals(
        self, 
        symbol: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for a symbol.
        
        Note: Screener.in provides current + historical quarterly data.
        We'll fetch all available and let the caller filter by date.
        
        Args:
            symbol: NSE ticker (e.g., 'RELIANCE')
            start_date: Not used (fetch all available)
            end_date: Not used (fetch all available)
            
        Returns:
            DataFrame with fundamental data
        """
        # Check cache first (skip date check - use all cached data)
        cache_file = self.cache_dir / f"{symbol}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df['date'] = pd.to_datetime(df['date'])
                # Cache is less than 1 day old - use it
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age.days < 1:
                    return df  # Return all data, no date filtering
            except Exception:
                pass  # Cache corrupted, refetch
        
        # Fetch from Screener.in
        try:
            # Rate limit
            self._rate_limit_wait()
            
            # Fetch consolidated data
            url = f"{self.base_url}/{symbol}/consolidated/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"   Failed to fetch {symbol}: HTTP {response.status_code}")
                return pd.DataFrame()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract current fundamentals and quarterly data
            fundamentals = self._extract_fundamentals(symbol, soup)
            
            if not fundamentals:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(fundamentals)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Apply 45-day reporting delay
            df = self._apply_reporting_delay(df, delay_days=45)
            
            # Cache results
            if not df.empty:
                df.to_parquet(cache_file, index=False)
            
            # Return all data - let caller handle date filtering if needed
            return df
            
        except Exception as e:
            print(f"   Error fetching {symbol} from Screener.in: {e}")
            return pd.DataFrame()
    
    def _extract_fundamentals(self, symbol: str, soup: BeautifulSoup) -> List[Dict]:
        """Extract fundamental metrics from Screener.in page."""
        
        records = []
        
        try:
            # Extract quarterly results table
            quarterly_table = soup.find('section', {'id': 'quarters'})
            if quarterly_table:
                records.extend(self._parse_quarterly_table(symbol, quarterly_table))
            
            # If no quarterly data, extract current ratios at least
            if not records:
                current_data = self._extract_current_ratios(symbol, soup)
                if current_data:
                    records.append(current_data)
            
        except Exception as e:
            print(f"   Error parsing {symbol}: {e}")
        
        return records
    
    def _parse_quarterly_table(self, symbol: str, section) -> List[Dict]:
        """Parse quarterly results table."""
        records = []
        
        try:
            table = section.find('table', {'class': 'data-table'})
            if not table:
                return records
            
            # Get headers (dates)
            headers = table.find('thead')
            if not headers:
                return records
            
            dates = []
            for th in headers.find_all('th')[1:]:  # Skip first column (metric names)
                date_text = th.text.strip()
                try:
                    # Parse date formats like "Mar 2024", "Jun 2023", etc.
                    date_obj = pd.to_datetime(date_text, format='%b %Y')
                    # Set to end of quarter
                    if date_obj.month in [3]:  # Mar
                        date_obj = date_obj.replace(day=31)
                    elif date_obj.month in [6]:  # Jun
                        date_obj = date_obj.replace(day=30)
                    elif date_obj.month in [9]:  # Sep
                        date_obj = date_obj.replace(day=30)
                    elif date_obj.month in [12]:  # Dec
                        date_obj = date_obj.replace(day=31)
                    dates.append(date_obj)
                except Exception:
                    continue
            
            if not dates:
                return records
            
            # Initialize records for each quarter
            for date_obj in dates:
                records.append({
                    'symbol': symbol,
                    'date': date_obj,
                    'report_date': date_obj,
                })
            
            # Parse rows
            tbody = table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    cols = row.find_all('td')
                    if len(cols) < 2:
                        continue
                    
                    metric_name = cols[0].text.strip().lower()
                    values = [col.text.strip() for col in cols[1:]]
                    
                    # Map metric names to our schema
                    field_mapping = {
                        'sales': 'revenue',
                        'operating profit': 'operating_profit',
                        'net profit': 'net_income',
                        'eps in rs': 'eps',
                    }
                    
                    field_name = field_mapping.get(metric_name)
                    if field_name:
                        for i, value in enumerate(values[:len(records)]):
                            try:
                                # Clean value (remove commas, convert to float)
                                cleaned = value.replace(',', '').strip()
                                if cleaned and cleaned != '-':
                                    records[i][field_name] = float(cleaned)
                            except (ValueError, AttributeError):
                                pass
            
        except Exception as e:
            print(f"   Error parsing quarterly table: {e}")
        
        # Now extract ratios from the page for the latest quarter
        if records:
            latest = records[0]
            self._add_ratios_to_record(latest, section.find_parent())
        
        return records
    
    def _extract_current_ratios(self, symbol: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract current fundamental ratios from the page."""
        record = {
            'symbol': symbol,
            'date': pd.Timestamp.now(),
            'report_date': pd.Timestamp.now(),
        }
        
        self._add_ratios_to_record(record, soup)
        
        return record if len(record) > 3 else None
    
    def _add_ratios_to_record(self, record: Dict, soup):
        """Add fundamental ratios to a record by scraping the page."""
        try:
            # Find all list items with ratios
            ratio_lis = soup.find_all('li', {'class': 'flex flex-space-between'})
            
            for li in ratio_lis:
                name_span = li.find('span', {'class': 'name'})
                value_span = li.find('span', {'class': 'number'})
                
                if not name_span or not value_span:
                    continue
                
                metric_name = name_span.text.strip().lower()
                value_text = value_span.text.strip()
                
                # Map Screener.in metrics to our schema
                metric_mapping = {
                    'market cap': 'market_cap',
                    'stock p/e': 'pe_ratio',
                    'book value': 'book_value_per_share',
                    'dividend yield': 'dividend_yield',
                    'roce': 'roce',
                    'roe': 'roe',
                    'face value': 'face_value',
                    'price to book value': 'pb_ratio',
                    'debt to equity': 'debt_to_equity',
                    'current ratio': 'current_ratio',
                    'sales growth': 'sales_growth',
                    'profit growth': 'profit_growth',
                    'return on equity': 'roe',
                }
                
                field_name = metric_mapping.get(metric_name)
                if field_name:
                    try:
                        # Clean value - handle different formats
                        cleaned = value_text.replace(',', '').replace('%', '').strip()
                        
                        # Handle special cases
                        if 'Cr.' in cleaned:  # Crores
                            cleaned = cleaned.replace('Cr.', '').strip()
                            record[field_name] = float(cleaned) * 10000000  # Convert to actual value
                        elif cleaned and cleaned not in ['-', 'N/A']:
                            record[field_name] = float(cleaned)
                    except (ValueError, AttributeError):
                        pass
        
        except Exception as e:
            print(f"   Error extracting ratios: {e}")
    
    def _apply_reporting_delay(self, df: pd.DataFrame, delay_days: int = 45) -> pd.DataFrame:
        """Apply 45-day reporting delay for point-in-time correctness."""
        if df.empty or 'report_date' not in df.columns:
            return df
        
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Shift availability forward by delay
        df['available_date'] = df['report_date'] + pd.Timedelta(days=delay_days)
        
        # For quarterly data, we forward-fill from available_date
        # Create daily records from each quarter's available_date
        
        return df
    
    def is_available(self) -> bool:
        """Check if Screener.in is accessible."""
        try:
            response = self.session.get(self.base_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def get_screener_provider(cache_dir: Optional[Path] = None) -> ScreenerInProvider:
    """Get configured Screener.in provider."""
    return ScreenerInProvider(cache_dir=cache_dir)


if __name__ == "__main__":
    # Test the provider
    provider = get_screener_provider()
    
    # Test with Reliance
    print("Testing Screener.in provider with RELIANCE...")
    start = datetime(2020, 1, 1)
    end = datetime(2024, 12, 31)
    
    df = provider.fetch_fundamentals("RELIANCE", start, end)
    
    if not df.empty:
        print(f"\n✅ Successfully fetched {len(df)} records")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head())
    else:
        print("\n❌ Failed to fetch data")
