"""
Scrape historical fundamental data from Screener.in for NIFTY 100 stocks.
Based on GitHub projects: jgera/screener.in and ketanmukadam/StockData

This is a POC to demonstrate the pipeline works before committing to paid APIs.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from datetime import datetime
import json

class ScreenerScraper:
    """Scraper for Screener.in fundamental data."""
    
    def __init__(self, cache_dir="data/raw/screener_cache"):
        self.base_url = "https://www.screener.in"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Headers to mimic browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
    
    def fetch_company_data(self, ticker):
        """Fetch fundamental data for a single company."""
        # Check cache first
        cache_file = self.cache_dir / f"{ticker}.json"
        if cache_file.exists():
            print(f"  📦 {ticker}: Using cached data")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        try:
            # Screener.in URL format
            url = f"{self.base_url}/company/{ticker}/consolidated/"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract quarterly results table
            data = self._parse_quarterly_results(soup, ticker)
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  ✅ {ticker}: Fetched {len(data)} quarters")
            return data
            
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")
            return []
    
    def _parse_quarterly_results(self, soup, ticker):
        """Parse the quarterly results table from Screener.in."""
        results = []
        
        try:
            # Find the quarterly results section
            section = soup.find('section', {'id': 'quarters'})
            if not section:
                return results
            
            table = section.find('table')
            if not table:
                return results
            
            # Get headers (dates)
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th')[1:]:  # Skip first column (metric name)
                date_str = th.text.strip()
                # Convert "Mar 2023" to "2023-03-31"
                try:
                    date_obj = pd.to_datetime(date_str, format='%b %Y')
                    # Set to last day of quarter
                    if date_obj.month == 3:
                        date_obj = date_obj.replace(day=31)
                    elif date_obj.month == 6:
                        date_obj = date_obj.replace(day=30)
                    elif date_obj.month == 9:
                        date_obj = date_obj.replace(day=30)
                    elif date_obj.month == 12:
                        date_obj = date_obj.replace(day=31)
                    headers.append(date_obj)
                except:
                    continue
            
            # Parse rows
            tbody = table.find('tbody')
            rows = tbody.find_all('tr')
            
            # Initialize data structure
            for i, date in enumerate(headers):
                results.append({
                    'ticker': ticker,
                    'date': date.strftime('%Y-%m-%d'),
                    'report_date': date.strftime('%Y-%m-%d'),
                })
            
            # Extract metrics
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 2:
                    continue
                
                metric_name = cells[0].text.strip().lower()
                
                # Map Screener.in metrics to our schema
                metric_map = {
                    'sales': 'revenue',
                    'operating profit': 'operating_profit',
                    'net profit': 'net_income',
                    'eps in rs': 'eps',
                }
                
                if metric_name not in metric_map:
                    continue
                
                our_metric = metric_map[metric_name]
                
                # Extract values for each quarter
                for i, cell in enumerate(cells[1:]):
                    if i >= len(results):
                        break
                    
                    value_str = cell.text.strip().replace(',', '')
                    try:
                        value = float(value_str)
                        results[i][our_metric] = value
                    except:
                        results[i][our_metric] = None
            
            return results
            
        except Exception as e:
            print(f"    Parse error: {e}")
            return results

def main():
    print("=" * 60)
    print("Screener.in Fundamental Data Scraper (POC)")
    print("=" * 60)
    
    # Load NIFTY 100 tickers
    universe_path = Path("data/external/universe/nifty100_universe.csv")
    universe = pd.read_csv(universe_path)
    tickers = universe['ticker'].dropna().tolist()  # Full NIFTY 100
    
    print(f"\n[1/3] Scraping fundamentals for {len(tickers)} tickers...")
    print("Rate limited to 1 request every 3 seconds to avoid bans.")
    print(f"Estimated time: ~{len(tickers) * 3 / 60:.1f} minutes\n")
    
    scraper = ScreenerScraper()
    all_data = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        data = scraper.fetch_company_data(ticker)
        all_data.extend(data)
        
        # Rate limiting
        if i < len(tickers):
            time.sleep(3)
    
    # Convert to DataFrame
    print(f"\n[2/3] Processing {len(all_data)} records...")
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("❌ No data scraped!")
        return
    
    # Add available_date (reporting delay)
    df['date'] = pd.to_datetime(df['date'])
    df['report_date'] = pd.to_datetime(df['report_date'])
    df['available_date'] = df['report_date'] + pd.Timedelta(days=45)
    
    # Save
    output_path = Path("data/processed/fundamentals_poc.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\n[3/3] Saved to {output_path}")
    print(f"\n✅ SUCCESS!")
    print(f"Total records: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    main()
