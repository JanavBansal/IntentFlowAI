"""
NSE Options Data Provider

Fetches options data from NSE India for:
- Put-Call Ratio (PCR)
- Open Interest (OI) analysis
- Max Pain calculation
- IV Rank

These sentiment indicators are useful for equity trading decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptionsConfig:
    """Configuration for NSE options provider."""
    
    # Cache directory
    cache_dir: Path = Path("data/cache/options")
    
    # NSE API base URL
    nse_base_url: str = "https://www.nseindia.com"
    
    # Default index for market-wide PCR
    default_index: str = "NIFTY"
    
    # Cache duration in hours
    cache_hours: int = 4


class NSEOptionsProvider:
    """
    Provide NSE options data for trading signals.
    
    Note: NSE API has rate limits and may require headers/cookies.
    This implementation provides the structure; actual fetching
    may need browser automation or API key access.
    
    Usage:
        provider = NSEOptionsProvider()
        
        # Get market-wide PCR
        pcr_data = provider.get_pcr("NIFTY")
        
        # Get stock PCR (for F&O stocks)
        stock_pcr = provider.get_pcr("RELIANCE")
    """
    
    # F&O stocks in NIFTY universe
    FNO_STOCKS = {
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
        "BAJFINANCE", "SUNPHARMA", "WIPRO", "HCLTECH", "TATAMOTORS",
        "ULTRACEMCO", "TATASTEEL", "NTPC", "POWERGRID", "TECHM",
        "ADANIENT", "ADANIPORTS", "ONGC", "COALINDIA", "JSWSTEEL",
        "M&M", "NESTLEIND", "BAJAJFINSV", "DRREDDY", "DIVISLAB",
        "CIPLA", "BRITANNIA", "APOLLOHOSP", "EICHERMOT", "HEROMOTOCO",
        "HINDALCO", "GRASIM", "BPCL", "INDUSINDBK", "SBILIFE",
        "HDFC", "HDFCLIFE", "UPL", "TATACONSUM", "BAJAJ-AUTO",
    }
    
    def __init__(self, config: Optional[OptionsConfig] = None):
        self.config = config or OptionsConfig()
        self._session = None
    
    def get_pcr(self, symbol: str = "NIFTY") -> Dict[str, Any]:
        """
        Get Put-Call Ratio and related metrics.
        
        Args:
            symbol: Index or stock symbol
            
        Returns:
            Dictionary with PCR metrics
        """
        result = {
            "symbol": symbol,
            "pcr": np.nan,
            "pcr_volume": np.nan,
            "max_pain": np.nan,
            "iv_rank": np.nan,
            "total_ce_oi": np.nan,
            "total_pe_oi": np.nan,
        }
        
        # Try to load from cache first
        cached = self._load_from_cache(symbol)
        if cached is not None:
            return cached
        
        # Try to fetch from NSE
        try:
            option_chain = self._fetch_option_chain(symbol)
            if option_chain:
                result = self._parse_option_chain(symbol, option_chain)
                self._save_to_cache(symbol, result)
        except Exception as e:
            logger.warning(f"Error fetching options data for {symbol}: {e}")
        
        return result
    
    def get_options_features(self, date: datetime | str) -> Dict[str, float]:
        """
        Get options-based features for all relevant indices/stocks.
        
        Args:
            date: Date for features (used for cache lookup)
            
        Returns:
            Dictionary of options features
        """
        features = {}
        
        # Market-wide PCR from NIFTY
        nifty_pcr = self.get_pcr("NIFTY")
        features["nifty_pcr"] = nifty_pcr.get("pcr", np.nan)
        features["nifty_pcr_volume"] = nifty_pcr.get("pcr_volume", np.nan)
        features["nifty_max_pain"] = nifty_pcr.get("max_pain", np.nan)
        
        # Bank Nifty PCR (additional sentiment)
        banknifty_pcr = self.get_pcr("BANKNIFTY")
        features["banknifty_pcr"] = banknifty_pcr.get("pcr", np.nan)
        
        # Compute PCR z-score (vs historical)
        pcr = features["nifty_pcr"]
        if not np.isnan(pcr):
            features["nifty_pcr_zscore"] = self._compute_pcr_zscore(pcr)
        else:
            features["nifty_pcr_zscore"] = np.nan
        
        return features
    
    def get_stock_pcr(self, ticker: str) -> Optional[float]:
        """
        Get PCR for individual F&O stock.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            PCR value or None if not F&O stock
        """
        if ticker not in self.FNO_STOCKS:
            return None
        
        data = self.get_pcr(ticker)
        return data.get("pcr")
    
    def is_fno_stock(self, ticker: str) -> bool:
        """Check if ticker is in F&O segment."""
        return ticker in self.FNO_STOCKS
    
    def _fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """
        Fetch option chain from NSE API.
        
        Note: This requires proper headers and session management.
        For production use, consider:
        1. Using official NSE API with credentials
        2. Using a third-party data provider
        3. Caching data locally
        """
        # Check if we have cached data
        cache_file = self._get_cache_file(symbol)
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        # For production, implement actual NSE API call here
        # The NSE API requires specific headers and session cookies
        # Example endpoint:
        # GET https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY
        # GET https://www.nseindia.com/api/option-chain-equities?symbol=RELIANCE
        
        logger.debug(f"Option chain fetch not implemented for {symbol}")
        return None
    
    def _parse_option_chain(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Parse NSE option chain data."""
        result = {
            "symbol": symbol,
            "pcr": np.nan,
            "pcr_volume": np.nan,
            "max_pain": np.nan,
            "iv_rank": np.nan,
            "total_ce_oi": np.nan,
            "total_pe_oi": np.nan,
        }
        
        try:
            records = data.get("records", {})
            chain_data = records.get("data", [])
            
            if not chain_data:
                return result
            
            total_ce_oi = 0
            total_pe_oi = 0
            total_ce_volume = 0
            total_pe_volume = 0
            
            for strike_data in chain_data:
                # Call option
                ce = strike_data.get("CE", {})
                if ce:
                    total_ce_oi += ce.get("openInterest", 0)
                    total_ce_volume += ce.get("totalTradedVolume", 0)
                
                # Put option
                pe = strike_data.get("PE", {})
                if pe:
                    total_pe_oi += pe.get("openInterest", 0)
                    total_pe_volume += pe.get("totalTradedVolume", 0)
            
            # Calculate PCR
            if total_ce_oi > 0:
                result["pcr"] = total_pe_oi / total_ce_oi
            
            if total_ce_volume > 0:
                result["pcr_volume"] = total_pe_volume / total_ce_volume
            
            result["total_ce_oi"] = total_ce_oi
            result["total_pe_oi"] = total_pe_oi
            
            # Calculate max pain
            result["max_pain"] = self._calculate_max_pain(chain_data)
            
        except Exception as e:
            logger.warning(f"Error parsing option chain for {symbol}: {e}")
        
        return result
    
    def _calculate_max_pain(self, chain_data: List[Dict]) -> Optional[float]:
        """
        Calculate max pain strike price.
        
        Max pain is the strike at which option writers have minimum loss.
        """
        if not chain_data:
            return None
        
        strikes = []
        
        for strike_data in chain_data:
            strike = strike_data.get("strikePrice")
            if strike is None:
                continue
            
            ce = strike_data.get("CE", {})
            pe = strike_data.get("PE", {})
            
            ce_oi = ce.get("openInterest", 0)
            pe_oi = pe.get("openInterest", 0)
            
            strikes.append({
                "strike": strike,
                "ce_oi": ce_oi,
                "pe_oi": pe_oi,
            })
        
        if not strikes:
            return None
        
        # Calculate pain at each strike
        min_pain = float("inf")
        max_pain_strike = None
        
        for s in strikes:
            strike = s["strike"]
            pain = 0
            
            for other in strikes:
                other_strike = other["strike"]
                
                # CE holders lose if spot > strike
                if other_strike < strike:
                    pain += other["ce_oi"] * (strike - other_strike)
                
                # PE holders lose if spot < strike
                if other_strike > strike:
                    pain += other["pe_oi"] * (other_strike - strike)
            
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike
        
        return max_pain_strike
    
    def _compute_pcr_zscore(self, current_pcr: float, lookback: int = 20) -> float:
        """
        Compute z-score of current PCR vs historical.
        
        High z-score (>1) = unusually high put buying = contrarian bullish
        Low z-score (<-1) = unusually high call buying = contrarian bearish
        """
        # For now, use static historical estimates
        # In production, maintain a rolling history
        historical_mean = 1.0
        historical_std = 0.3
        
        if historical_std > 0:
            return (current_pcr - historical_mean) / historical_std
        return 0.0
    
    def _get_cache_file(self, symbol: str) -> Path:
        """Get cache file path for symbol."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{symbol}_options.json"
    
    def _load_from_cache(self, symbol: str) -> Optional[Dict]:
        """Load options data from cache if fresh."""
        cache_file = self._get_cache_file(symbol)
        
        if not cache_file.exists():
            return None
        
        # Check if cache is fresh
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=self.config.cache_hours):
            return None
        
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, symbol: str, data: Dict) -> None:
        """Save options data to cache."""
        cache_file = self._get_cache_file(symbol)
        
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error saving options cache for {symbol}: {e}")


def get_nse_options_provider() -> NSEOptionsProvider:
    """Get configured NSE options provider."""
    return NSEOptionsProvider()


def compute_options_features(
    date: str | datetime,
    provider: Optional[NSEOptionsProvider] = None,
) -> Dict[str, float]:
    """
    Convenience function to compute options features.
    
    Args:
        date: Date for features
        provider: Optional provider instance
        
    Returns:
        Dictionary of options features
    """
    if provider is None:
        provider = get_nse_options_provider()
    return provider.get_options_features(date)
