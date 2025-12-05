"""
Indian Stock News Scraper

Scrapes financial news for NIFTY tickers from multiple sources:
1. Yahoo Finance (yfinance) - Primary source, ticker-specific
2. Google News RSS - Fallback for additional coverage

Output: CSV with columns [date, ticker, title, source, url, reliability]
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import feedparser

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")


def scrape_ticker_news_yfinance(ticker: str, max_articles: int = 50):
    """Scrape news for a single ticker from Yahoo Finance."""
    if not HAS_YFINANCE:
        return []
    
    articles = []
    
    try:
        # Add .NS suffix for NSE stocks
        yf_ticker = yf.Ticker(f"{ticker}.NS")
        news_items = yf_ticker.news
        
        if not news_items:
            return []
        
        for item in news_items[:max_articles]:
            try:
                # Extract timestamp (providerPublishTime is Unix timestamp)
                pub_time = item.get('providerPublishTime')
                if pub_time:
                    date = datetime.fromtimestamp(pub_time)
                else:
                    date = datetime.now()
                
                articles.append({
                    'date': date,
                    'ticker': ticker,
                    'title': item.get('title', ''),
                    'source': item.get('publisher', 'yahoo_finance'),
                    'url': item.get('link', ''),
                    'reliability': 0.8  # Yahoo Finance is reliable
                })
            except Exception as e:
                print(f"  Warning: Failed to parse article for {ticker}: {e}")
                continue
                
    except Exception as e:
        print(f"  Error fetching Yahoo news for {ticker}: {e}")
    
    return articles


def scrape_ticker_news_google(ticker: str, max_articles: int = 10):
    """Scrape news for a single ticker from Google News RSS."""
    articles = []
    
    try:
        # Google News RSS search URL
        query = f"{ticker} stock India"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(url)
        
        if not feed.entries:
            return []
        
        for entry in feed.entries[:max_articles]:
            try:
                # Parse published date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    date = datetime(*entry.published_parsed[:6])
                else:
                    date = datetime.now()
                
                articles.append({
                    'date': date,
                    'ticker': ticker,
                    'title': entry.title,
                    'source': 'google_news',
                    'url': entry.link,
                    'reliability': 0.6  # Google News is less filtered
                })
            except Exception as e:
                print(f"  Warning: Failed to parse Google article for {ticker}: {e}")
                continue
                
    except Exception as e:
        print(f"  Error fetching Google news for {ticker}: {e}")
    
    return articles


def scrape_ticker_news(ticker: str, use_google: bool = True):
    """Scrape news for a single ticker from all sources."""
    all_articles = []
    
    # Primary: Yahoo Finance
    yf_articles = scrape_ticker_news_yfinance(ticker)
    all_articles.extend(yf_articles)
    
    # Secondary: Google News (if enabled and Yahoo returned < 5 articles)
    if use_google and len(yf_articles) < 5:
        google_articles = scrape_ticker_news_google(ticker)
        all_articles.extend(google_articles)
    
    return all_articles


def scrape_universe_news(universe_file: str, output_file: str, 
                         max_tickers: int = None, use_google: bool = False):
    """
    Scrape news for all tickers in the universe.
    
    Args:
        universe_file: Path to CSV with 'ticker' column
        output_file: Path to save output CSV
        max_tickers: If set, only scrape first N tickers (for testing)
        use_google: Whether to use Google News as fallback
    """
    # Load universe
    universe = pd.read_csv(universe_file)
    
    if 'ticker' not in universe.columns:
        raise ValueError(f"Universe file must have 'ticker' column. Found: {universe.columns.tolist()}")
    
    tickers = universe['ticker'].unique()
    
    if max_tickers:
        tickers = tickers[:max_tickers]
        print(f"Testing mode: Scraping only first {max_tickers} tickers")
    
    print(f"Scraping news for {len(tickers)} tickers...")
    
    all_news = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Scraping {ticker}...", end=' ')
        
        try:
            articles = scrape_ticker_news(ticker, use_google=use_google)
            
            if articles:
                all_news.extend(articles)
                print(f"✅ {len(articles)} articles")
            else:
                print("⚠️  No articles found")
                failed_tickers.append(ticker)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_tickers.append(ticker)
        
        # Rate limiting: 1 second between tickers
        if i < len(tickers):
            time.sleep(1)
    
    # Convert to DataFrame
    if all_news:
        df = pd.DataFrame(all_news)
        
        # Sort by date (most recent first)
        df = df.sort_values('date', ascending=False)
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Scraped {len(df)} articles for {len(df['ticker'].unique())} tickers")
        print(f"   Saved to: {output_path}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        if failed_tickers:
            print(f"\n⚠️  Failed to scrape {len(failed_tickers)} tickers:")
            print(f"   {', '.join(failed_tickers[:10])}" + (" ..." if len(failed_tickers) > 10 else ""))
        
        return df
    else:
        print("\n❌ No articles scraped. Check your internet connection and ticker symbols.")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Indian stock news")
    parser.add_argument("--universe", default="data/external/universe/nifty464_universe.csv",
                        help="Path to universe CSV file")
    parser.add_argument("--output", default="data/raw/indian_ticker_news.csv",
                        help="Path to save output CSV")
    parser.add_argument("--test", type=int, default=None,
                        help="Test mode: scrape only N tickers")
    parser.add_argument("--google", action="store_true",
                        help="Enable Google News as fallback")
    
    args = parser.parse_args()
    
    scrape_universe_news(
        universe_file=args.universe,
        output_file=args.output,
        max_tickers=args.test,
        use_google=args.google
    )
