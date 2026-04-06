"""
Macro Data Provider

Provides macro-economic indicators for feature engineering:
- India VIX (fear gauge)
- USD/INR exchange rate
- Crude oil prices
- US 10Y Treasury yield
- FII/DII flow data

These affect all stocks and provide important market regime context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MacroConfig:
    """Configuration for macro data provider."""

    # Data directories
    cache_dir: Path = Path("data/cache/macro")

    # Yahoo Finance tickers for macro data
    vix_ticker: str = "^INDIAVIX"  # India VIX
    usdinr_ticker: str = "USDINR=X"  # USD/INR
    crude_ticker: str = "CL=F"  # Crude oil futures
    us10y_ticker: str = "^TNX"  # US 10Y yield
    nifty_ticker: str = "^NSEI"  # NIFTY 50

    # Global market tickers (overnight returns for India open prediction)
    sp500_ticker: str = "^GSPC"       # S&P 500
    nasdaq_ticker: str = "^IXIC"      # Nasdaq Composite
    nikkei_ticker: str = "^N225"      # Nikkei 225
    hangseng_ticker: str = "^HSI"     # Hang Seng Index
    gold_ticker: str = "GC=F"        # Gold futures
    copper_ticker: str = "HG=F"      # Copper futures (growth proxy)
    niftybank_ticker: str = "^NSEBANK"  # NIFTY Bank
    niftyit_ticker: str = "^CNXIT"    # NIFTY IT

    # FII/DII data file (CSV legacy path)
    fiidii_file: Optional[str] = None
    # FII/DII parquet cache (from fetch_fii_dii_data.py)
    fiidii_parquet: Path = Path("data/raw/fii_dii/fii_dii_cache.parquet")


class MacroDataProvider:
    """
    Provide macro-economic indicators for trading models.
    
    Usage:
        provider = MacroDataProvider()
        
        # Get all macro features for a date
        features = provider.get_macro_features(date)
        
        # Get macro DataFrame for date range
        df = provider.get_macro_df(start_date, end_date)
    """
    
    def __init__(self, config: Optional[MacroConfig] = None):
        self.config = config or MacroConfig()
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_macro_features(self, date: datetime | str | pd.Timestamp) -> Dict[str, float]:
        """
        Get all macro features for a specific date.
        
        Args:
            date: Date to get features for
            
        Returns:
            Dictionary of macro features
        """
        date = pd.to_datetime(date)
        
        features = {}
        
        # Get VIX
        vix = self._get_vix(date)
        if vix is not None:
            features["vix_india"] = vix
            features["vix_regime"] = self._classify_vix_regime(vix)
        
        # Get USD/INR
        usdinr = self._get_usdinr(date)
        if usdinr is not None:
            features["usd_inr"] = usdinr
            features["usd_inr_trend"] = self._get_trend(
                self.config.usdinr_ticker, date, lookback=20
            )
        
        # Get Crude oil
        crude = self._get_crude(date)
        if crude is not None:
            features["crude_oil"] = crude
            features["crude_trend"] = self._get_trend(
                self.config.crude_ticker, date, lookback=20
            )
        
        # Get US 10Y yield
        us10y = self._get_us10y(date)
        if us10y is not None:
            features["us_10y_yield"] = us10y
            features["yield_change_20d"] = self._get_change(
                self.config.us10y_ticker, date, lookback=20
            )
        
        # Get FII/DII flows
        fii_dii = self._get_fiidii_flows(date)
        features.update(fii_dii)
        
        # Market regime from NIFTY
        nifty_regime = self._get_nifty_regime(date)
        features.update(nifty_regime)

        # Global market overnight returns
        global_feats = self.get_global_features(date)
        features.update(global_feats)

        return features
    
    def get_macro_df(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Get macro data for a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            DataFrame with macro features per date
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate business days
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        
        records = []
        for date in dates:
            features = self.get_macro_features(date)
            features["date"] = date
            records.append(features)
        
        df = pd.DataFrame(records)
        df = df.set_index("date")
        
        return df
    
    def _get_vix(self, date: pd.Timestamp) -> Optional[float]:
        """Get India VIX value."""
        return self._get_price(self.config.vix_ticker, date)
    
    def _get_usdinr(self, date: pd.Timestamp) -> Optional[float]:
        """Get USD/INR exchange rate."""
        return self._get_price(self.config.usdinr_ticker, date)
    
    def _get_crude(self, date: pd.Timestamp) -> Optional[float]:
        """Get crude oil price."""
        return self._get_price(self.config.crude_ticker, date)
    
    def _get_us10y(self, date: pd.Timestamp) -> Optional[float]:
        """Get US 10Y Treasury yield."""
        return self._get_price(self.config.us10y_ticker, date)
    
    def _get_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Get closing price for a ticker on a date."""
        df = self._load_ticker_data(ticker)
        if df is None or df.empty:
            return None
        
        # Find closest date <= requested date
        df = df[df.index <= date]
        if df.empty:
            return None
        
        return float(df["close"].iloc[-1])
    
    def _get_trend(self, ticker: str, date: pd.Timestamp, lookback: int = 20) -> Optional[float]:
        """Calculate trend (return) over lookback period."""
        df = self._load_ticker_data(ticker)
        if df is None or df.empty:
            return None
        
        df = df[df.index <= date].tail(lookback + 1)
        if len(df) < 2:
            return None
        
        return float((df["close"].iloc[-1] / df["close"].iloc[0]) - 1)
    
    def _get_change(self, ticker: str, date: pd.Timestamp, lookback: int = 20) -> Optional[float]:
        """Calculate absolute change over lookback period."""
        df = self._load_ticker_data(ticker)
        if df is None or df.empty:
            return None
        
        df = df[df.index <= date].tail(lookback + 1)
        if len(df) < 2:
            return None
        
        return float(df["close"].iloc[-1] - df["close"].iloc[0])
    
    def _classify_vix_regime(self, vix: float) -> int:
        """
        Classify VIX into regime.
        
        Returns:
            0: Low volatility (VIX < 15)
            1: Normal volatility (15 <= VIX < 25)
            2: High volatility (VIX >= 25)
        """
        if vix < 15:
            return 0
        elif vix < 25:
            return 1
        else:
            return 2
    
    def _get_fiidii_flows(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Get FII/DII flow data from parquet cache or CSV fallback.

        Returns:
            Dictionary with flow metrics
        """
        result = {
            "fii_net_flow_5d": np.nan,
            "dii_net_flow_5d": np.nan,
            "fii_dii_ratio": np.nan,
        }

        df = self._load_fiidii_data()
        if df is None or df.empty:
            return result

        try:
            # Get 5-day window up to requested date
            window = df[df.index <= date].tail(5)

            if not window.empty:
                fii_col = "fii_cash_net" if "fii_cash_net" in df.columns else "fii_net"
                dii_col = "dii_cash_net" if "dii_cash_net" in df.columns else "dii_net"

                if fii_col in df.columns:
                    result["fii_net_flow_5d"] = float(window[fii_col].sum())
                if dii_col in df.columns:
                    result["dii_net_flow_5d"] = float(window[dii_col].sum())

                fii = result["fii_net_flow_5d"]
                dii = result["dii_net_flow_5d"]
                if not np.isnan(fii) and not np.isnan(dii) and dii != 0:
                    result["fii_dii_ratio"] = fii / abs(dii)

        except Exception as e:
            logger.debug(f"Error computing FII/DII flows: {e}")

        return result

    def _load_fiidii_data(self) -> Optional[pd.DataFrame]:
        """Load FII/DII data from parquet cache (preferred) or CSV fallback."""
        if "_fiidii_df" in self.__dict__:
            return self.__dict__["_fiidii_df"]

        df = None

        # 1. Try parquet cache (from fetch_fii_dii_data.py)
        parquet_path = Path(self.config.fiidii_parquet)
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                logger.debug(f"Loaded FII/DII from parquet: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load FII/DII parquet: {e}")
                df = None

        # 2. Fallback to CSV
        if df is None:
            fiidii_file = self.config.fiidii_file
            if fiidii_file is None:
                for path in [
                    Path("data/external/fiidii.csv"),
                    Path("data/raw/ownership/fiidii.csv"),
                ]:
                    if path.exists():
                        fiidii_file = str(path)
                        break
            if fiidii_file and Path(fiidii_file).exists():
                try:
                    df = pd.read_csv(fiidii_file, parse_dates=["date"])
                    df = df.set_index("date").sort_index()
                except Exception as e:
                    logger.debug(f"Error loading FII/DII CSV: {e}")

        self.__dict__["_fiidii_df"] = df
        return df

    def get_global_features(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Get global market overnight return features.

        US and Asian markets close before India opens, making their returns
        a reliable short-term predictor of Indian market direction.
        """
        features = {}

        # S&P 500 previous day return
        sp500_ret = self._get_return(self.config.sp500_ticker, date, lookback=1)
        if sp500_ret is not None:
            features["sp500_ret_1d"] = sp500_ret

        # Nasdaq previous day return
        nasdaq_ret = self._get_return(self.config.nasdaq_ticker, date, lookback=1)
        if nasdaq_ret is not None:
            features["nasdaq_ret_1d"] = nasdaq_ret

        # Asia (Nikkei + Hang Seng average) previous day
        nikkei_ret = self._get_return(self.config.nikkei_ticker, date, lookback=1)
        hs_ret = self._get_return(self.config.hangseng_ticker, date, lookback=1)
        asia_rets = [r for r in [nikkei_ret, hs_ret] if r is not None]
        if asia_rets:
            features["asia_ret_1d"] = float(np.mean(asia_rets))

        # Gold 5-day return (safe haven indicator)
        gold_ret = self._get_return(self.config.gold_ticker, date, lookback=5)
        if gold_ret is not None:
            features["gold_ret_5d"] = gold_ret

        # Copper 5-day return (growth proxy)
        copper_ret = self._get_return(self.config.copper_ticker, date, lookback=5)
        if copper_ret is not None:
            features["copper_ret_5d"] = copper_ret

        # DXY proxy: inverse of USD/INR 5d change
        usdinr_ret = self._get_return(self.config.usdinr_ticker, date, lookback=5)
        if usdinr_ret is not None:
            features["dxy_ret_5d"] = usdinr_ret  # INR weakening = USD strengthening

        return features

    def _get_return(self, ticker: str, date: pd.Timestamp, lookback: int = 1) -> Optional[float]:
        """Calculate return over lookback trading days."""
        df = self._load_ticker_data(ticker)
        if df is None or df.empty:
            return None

        df = df[df.index <= date].tail(lookback + 1)
        if len(df) < 2:
            return None

        return float((df["close"].iloc[-1] / df["close"].iloc[0]) - 1)
    
    def _get_nifty_regime(self, date: pd.Timestamp) -> Dict[str, Any]:
        """
        Get NIFTY-based market regime features.
        """
        result = {
            "nifty_trend": np.nan,
            "nifty_vol_20d": np.nan,
            "nifty_above_200ma": np.nan,
        }
        
        df = self._load_ticker_data(self.config.nifty_ticker)
        if df is None or df.empty:
            return result
        
        df = df[df.index <= date]
        if len(df) < 200:
            return result
        
        # 20-day trend
        if len(df) >= 20:
            trend = (df["close"].iloc[-1] / df["close"].iloc[-20]) - 1
            result["nifty_trend"] = float(trend)
        
        # 20-day volatility
        if len(df) >= 20:
            returns = df["close"].pct_change().tail(20)
            result["nifty_vol_20d"] = float(returns.std() * np.sqrt(252))
        
        # Above 200-day MA
        ma200 = df["close"].rolling(200).mean().iloc[-1]
        result["nifty_above_200ma"] = 1.0 if df["close"].iloc[-1] > ma200 else 0.0
        
        return result
    
    def _load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker data from cache or fetch."""
        if ticker in self._cache:
            return self._cache[ticker]
        
        # Try to load from parquet cache
        cache_dir = Path(self.config.cache_dir)
        cache_file = cache_dir / f"{ticker.replace('^', '').replace('=', '_')}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                self._cache[ticker] = df
                return df
            except Exception as e:
                logger.warning(f"Error loading cached data for {ticker}: {e}")
        
        # Try to fetch from Yahoo Finance
        try:
            import yfinance as yf
            
            data = yf.download(
                ticker,
                start="2010-01-01",
                end=datetime.now().strftime("%Y-%m-%d"),
                progress=False,
            )
            
            if data.empty:
                return None
            
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            df = data[["Close"]].copy()
            df.columns = ["close"]
            
            # Cache it
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file)
            
            self._cache[ticker] = df
            return df
            
        except Exception as e:
            logger.warning(f"Error fetching data for {ticker}: {e}")
            return None
    
    def refresh_cache(self, tickers: Optional[List[str]] = None) -> None:
        """Refresh cached macro data from Yahoo Finance."""
        if tickers is None:
            tickers = [
                self.config.vix_ticker,
                self.config.usdinr_ticker,
                self.config.crude_ticker,
                self.config.us10y_ticker,
                self.config.nifty_ticker,
                # Global market tickers
                self.config.sp500_ticker,
                self.config.nasdaq_ticker,
                self.config.nikkei_ticker,
                self.config.hangseng_ticker,
                self.config.gold_ticker,
                self.config.copper_ticker,
                self.config.niftybank_ticker,
                self.config.niftyit_ticker,
            ]
        
        for ticker in tickers:
            # Clear from cache to force refresh
            self._cache.pop(ticker, None)
            
            # Remove cached file
            cache_dir = Path(self.config.cache_dir)
            cache_file = cache_dir / f"{ticker.replace('^', '').replace('=', '_')}.parquet"
            if cache_file.exists():
                cache_file.unlink()
            
            # Reload
            self._load_ticker_data(ticker)
        
        logger.info(f"Refreshed macro data cache for {len(tickers)} tickers")


def get_macro_provider() -> MacroDataProvider:
    """Get configured macro data provider."""
    return MacroDataProvider()


def compute_macro_features(
    date: str | datetime,
    provider: Optional[MacroDataProvider] = None,
) -> Dict[str, float]:
    """
    Convenience function to compute macro features for a date.
    
    Args:
        date: Date to compute features for
        provider: Optional provider instance
        
    Returns:
        Dictionary of macro features
    """
    if provider is None:
        provider = get_macro_provider()
    return provider.get_macro_features(date)
