"""
India VIX Data Provider

Fetches India VIX (volatility index) data for regime detection.
India VIX is the "fear gauge" for the Indian market.

Sources:
- Yahoo Finance: ^INDIAVIX
- NSE website (backup)

Usage:
    provider = IndiaVIXProvider()
    vix_data = provider.fetch(start_date="2020-01-01", end_date="2024-12-01")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndiaVIXProvider:
    """
    Fetches India VIX data for regime detection.
    
    Primary source: Yahoo Finance (^INDIAVIX)
    Fallback: Cached data or synthetic proxy
    """
    
    cache_dir: Path = Path("data/raw/vix")
    cache_file: str = "india_vix.parquet"
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self.cache_dir / self.cache_file
    
    def fetch(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch India VIX data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with columns: date, close (VIX level)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Try cache first
        if use_cache and self._cache_path.exists():
            cached = self._load_cache()
            if cached is not None and not cached.empty:
                logger.info(f"Loaded {len(cached)} rows of India VIX from cache")
                return self._filter_dates(cached, start_date, end_date)
        
        # Fetch from Yahoo Finance
        try:
            vix_data = self._fetch_from_yahoo(start_date, end_date)
            if not vix_data.empty:
                self._save_cache(vix_data)
                return vix_data
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed: {e}")
        
        # Fallback to cache even if not requested
        if self._cache_path.exists():
            cached = self._load_cache()
            if cached is not None:
                logger.warning("Using cached VIX data as fallback")
                return self._filter_dates(cached, start_date, end_date)
        
        # Last resort: return empty DataFrame
        logger.error("Could not fetch India VIX data from any source")
        return pd.DataFrame(columns=["date", "close"])
    
    def _fetch_from_yahoo(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch India VIX from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required. Install with: pip install yfinance")
        
        logger.info(f"Fetching India VIX from Yahoo Finance: {start_date} to {end_date}")
        
        ticker = yf.Ticker("^INDIAVIX")
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning("No data returned from Yahoo Finance for ^INDIAVIX")
            return pd.DataFrame(columns=["date", "close"])
        
        # Format output
        vix_data = pd.DataFrame({
            "date": hist.index.date,
            "close": hist["Close"].values,
            "high": hist["High"].values,
            "low": hist["Low"].values,
            "open": hist["Open"].values
        })
        vix_data["date"] = pd.to_datetime(vix_data["date"])
        
        logger.info(f"Fetched {len(vix_data)} days of India VIX data")
        return vix_data
    
    def _load_cache(self) -> Optional[pd.DataFrame]:
        """Load cached VIX data."""
        try:
            df = pd.read_parquet(self._cache_path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.warning(f"Failed to load VIX cache: {e}")
            return None
    
    def _save_cache(self, df: pd.DataFrame) -> None:
        """Save VIX data to cache."""
        try:
            df.to_parquet(self._cache_path, index=False)
            logger.info(f"Cached India VIX data to {self._cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save VIX cache: {e}")
    
    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Filter DataFrame to date range."""
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        return df.loc[mask].reset_index(drop=True)


@dataclass
class FIIDIIProvider:
    """
    Fetches FII/DII flow data from NSE.
    
    Data includes:
    - FII net buy/sell (cash + derivatives)
    - DII net buy/sell (cash + derivatives)
    """
    
    cache_dir: Path = Path("data/raw/flows")
    cache_file: str = "fii_dii_flows.parquet"
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self.cache_dir / self.cache_file
    
    def fetch(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch FII/DII flow data.
        
        Returns:
            DataFrame with columns: date, fii_net, dii_net, fii_cash, dii_cash
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # For now, return synthetic data based on market returns
        # In production, this would scrape NSE or use a data API
        logger.warning("FII/DII data currently uses synthetic proxy. Consider integrating NSE data.")
        
        return self._generate_synthetic_flows(start_date, end_date)
    
    def _generate_synthetic_flows(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate synthetic FII/DII flows for testing.
        
        In production, replace with actual NSE data scraping.
        """
        import numpy as np
        
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        
        # Generate correlated random flows
        np.random.seed(42)
        
        # FII flows: more volatile, sometimes negative
        fii_net = np.random.normal(500, 2000, n)  # Mean positive, high std
        
        # DII flows: often counterbalance FII
        dii_net = -0.5 * fii_net + np.random.normal(200, 1000, n)
        
        df = pd.DataFrame({
            "date": dates,
            "fii_net": fii_net,
            "dii_net": dii_net,
            "fii_cash": fii_net * 0.7,  # Approximate cash vs derivatives split
            "dii_cash": dii_net * 0.8
        })
        
        return df


def test_regime_detector():
    """Test the regime detector with sample data."""
    from intentflow_ai.modeling.regime_detector import (
        RegimeDetector, RegimeConfig, compute_regime_features
    )
    
    # Create sample price data
    dates = pd.date_range("2023-01-01", "2024-12-01", freq="B")
    n_dates = len(dates)
    n_tickers = 50
    
    import numpy as np
    np.random.seed(42)
    
    # Simulate prices
    prices_data = []
    for ticker in [f"STOCK{i}" for i in range(n_tickers)]:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, n_dates)))
        for i, d in enumerate(dates):
            prices_data.append({"date": d, "ticker": ticker, "close": prices[i]})
    
    prices_df = pd.DataFrame(prices_data)
    
    # Simulate VIX
    vix_df = pd.DataFrame({
        "date": dates,
        "close": 15 + 10 * np.abs(np.sin(np.linspace(0, 4*np.pi, n_dates))) + np.random.normal(0, 2, n_dates)
    })
    
    # Compute regime features
    features = compute_regime_features(prices_df, vix_df)
    print(f"Computed {len(features)} days of regime features")
    print(features.head())
    
    # Fit regime detector
    detector = RegimeDetector()
    detector.fit(features)
    
    # Predict current regime
    regime = detector.predict_regime(features.tail(30))
    print(f"\nCurrent regime: {regime}")
    print(f"Should trade: {detector.should_trade()}")
    print(f"Regime summary: {detector.get_regime_summary()}")


if __name__ == "__main__":
    test_regime_detector()
