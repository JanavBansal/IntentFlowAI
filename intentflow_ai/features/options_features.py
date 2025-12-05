"""
Options-Based Features

Generate features from options market data:
- Put-Call Ratio (PCR) for index and stocks
- PCR Z-scores for sentiment extremes
- Max pain distance
- Open interest buildup analysis

Options data provides valuable sentiment and flow information
that can predict equity moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptionsFeatureConfig:
    """Configuration for options features."""
    
    # PCR thresholds for sentiment classification
    pcr_bullish_threshold: float = 1.2   # PCR > 1.2 = contrarian bullish
    pcr_bearish_threshold: float = 0.7   # PCR < 0.7 = contrarian bearish
    
    # Z-score lookback for PCR normalization
    pcr_zscore_lookback: int = 20  # Days
    
    # Historical PCR statistics (for z-score when no history)
    historical_pcr_mean: float = 1.0
    historical_pcr_std: float = 0.3
    
    # Max pain calculation
    max_pain_distance_threshold: float = 0.03  # 3% from spot


# F&O stocks in Indian market (subset of NIFTY universe)
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
    "HDFCLIFE", "UPL", "TATACONSUM", "BAJAJ-AUTO",
    "SBICARD", "IRCTC", "PVR", "MUTHOOTFIN", "BIOCON",
    "ASHOKLEY", "BANDHANBNK", "BANKBARODA", "BEL", "BHEL",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "CONCOR",
    "CUMMINSIND", "DLF", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "GAIL", "GMRINFRA", "GODREJCP", "GODREJPROP", "HAVELLS",
    "IDFCFIRSTB", "IGL", "INDHOTEL", "INDUSTOWER", "JUBLFOOD",
    "L&TFH", "LICHSGFIN", "LUPIN", "MANAPPURAM", "MARICO",
    "METROPOLIS", "MGL", "MPHASIS", "MRF", "NAUKRI",
    "NMDC", "OFSS", "PAGEIND", "PEL", "PERSISTENT",
    "PETRONET", "PFC", "PIDILITIND", "PIIND", "PNB",
    "POLYCAB", "RAMCOCEM", "RBLBANK", "RECLTD", "SAIL",
    "SHREECEM", "SRF", "SYNGENE", "TATACOMM", "TATAELXSI",
    "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "VEDL",
    "VOLTAS", "ZEEL", "ZYDUSLIFE",
}


def is_fno_stock(ticker: str) -> bool:
    """Check if a ticker is in F&O segment."""
    return ticker.upper() in FNO_STOCKS


def compute_options_features(
    ticker: str,
    close_price: float,
    nifty_pcr: Optional[float] = None,
    banknifty_pcr: Optional[float] = None,
    stock_pcr: Optional[float] = None,
    max_pain: Optional[float] = None,
    pcr_history: Optional[List[float]] = None,
    config: Optional[OptionsFeatureConfig] = None,
) -> Dict[str, Any]:
    """
    Compute options-based features for a stock.
    
    Args:
        ticker: Stock ticker
        close_price: Current closing price
        nifty_pcr: NIFTY index PCR
        banknifty_pcr: Bank Nifty index PCR
        stock_pcr: Individual stock PCR (if F&O stock)
        max_pain: Max pain strike price
        pcr_history: Historical PCR values for z-score
        config: Feature configuration
        
    Returns:
        Dictionary of options features
    """
    config = config or OptionsFeatureConfig()
    features = {}
    
    # NIFTY PCR (market-wide sentiment)
    if nifty_pcr is not None:
        features["nifty_pcr"] = nifty_pcr
        features["nifty_pcr_zscore"] = _compute_pcr_zscore(
            nifty_pcr, pcr_history, config
        )
        features["nifty_pcr_sentiment"] = _classify_pcr_sentiment(nifty_pcr, config)
    else:
        features["nifty_pcr"] = np.nan
        features["nifty_pcr_zscore"] = np.nan
        features["nifty_pcr_sentiment"] = 0
    
    # Bank Nifty PCR (banking sector sentiment)
    if banknifty_pcr is not None:
        features["banknifty_pcr"] = banknifty_pcr
    else:
        features["banknifty_pcr"] = np.nan
    
    # Stock-specific PCR (for F&O stocks)
    if is_fno_stock(ticker) and stock_pcr is not None:
        features["stock_pcr"] = stock_pcr
        features["stock_pcr_zscore"] = _compute_pcr_zscore(
            stock_pcr, None, config
        )
        features["stock_pcr_sentiment"] = _classify_pcr_sentiment(stock_pcr, config)
    else:
        features["stock_pcr"] = np.nan
        features["stock_pcr_zscore"] = np.nan
        features["stock_pcr_sentiment"] = 0
    
    # F&O flag
    features["is_fno_stock"] = 1.0 if is_fno_stock(ticker) else 0.0
    
    # Max pain distance
    if max_pain is not None and close_price > 0:
        features["max_pain"] = max_pain
        features["max_pain_distance"] = (close_price - max_pain) / close_price
        features["above_max_pain"] = 1.0 if close_price > max_pain else 0.0
    else:
        features["max_pain"] = np.nan
        features["max_pain_distance"] = np.nan
        features["above_max_pain"] = np.nan
    
    # Composite options sentiment
    features["options_sentiment_composite"] = _compute_composite_sentiment(features)
    
    return features


def _compute_pcr_zscore(
    pcr: float,
    history: Optional[List[float]],
    config: OptionsFeatureConfig,
) -> float:
    """Compute z-score of current PCR vs historical."""
    if history and len(history) >= 5:
        mean = np.mean(history)
        std = np.std(history)
        if std > 0:
            return (pcr - mean) / std
    
    # Use configured historical stats
    if config.historical_pcr_std > 0:
        return (pcr - config.historical_pcr_mean) / config.historical_pcr_std
    
    return 0.0


def _classify_pcr_sentiment(
    pcr: float,
    config: OptionsFeatureConfig,
) -> int:
    """
    Classify PCR into sentiment signal.
    
    Returns:
        1: Contrarian bullish (high PCR = excessive puts)
        0: Neutral
        -1: Contrarian bearish (low PCR = excessive calls)
    """
    if pcr > config.pcr_bullish_threshold:
        return 1  # Bullish (contrarian)
    elif pcr < config.pcr_bearish_threshold:
        return -1  # Bearish (contrarian)
    return 0


def _compute_composite_sentiment(features: Dict[str, Any]) -> float:
    """
    Compute composite options sentiment score.
    
    Combines multiple sentiment indicators into single score (-1 to +1).
    """
    scores = []
    weights = []
    
    # NIFTY PCR sentiment (market-wide)
    nifty_sentiment = features.get("nifty_pcr_sentiment", 0)
    if nifty_sentiment != 0:
        scores.append(nifty_sentiment)
        weights.append(0.4)
    
    # Stock PCR sentiment (if available)
    stock_sentiment = features.get("stock_pcr_sentiment", 0)
    if stock_sentiment != 0 and features.get("is_fno_stock", 0) == 1:
        scores.append(stock_sentiment)
        weights.append(0.4)
    
    # Max pain signal
    above_max_pain = features.get("above_max_pain")
    if above_max_pain is not None and not np.isnan(above_max_pain):
        # Above max pain = bullish, below = bearish
        scores.append(1.0 if above_max_pain else -1.0)
        weights.append(0.2)
    
    if not scores:
        return 0.0
    
    # Weighted average
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        return sum(s * w for s, w in zip(scores, weights))
    
    return 0.0


def analyze_oi_buildup(
    current_oi: Dict[str, float],
    previous_oi: Dict[str, float],
    current_price: float,
    previous_price: float,
) -> Dict[str, Any]:
    """
    Analyze Open Interest buildup patterns.
    
    OI + Price patterns:
    - Long buildup: OI up + Price up (bullish)
    - Short buildup: OI up + Price down (bearish)
    - Long unwinding: OI down + Price down (bearish ending)
    - Short covering: OI down + Price up (bullish)
    
    Args:
        current_oi: Current OI data {"call": x, "put": y}
        previous_oi: Previous OI data
        current_price: Current price
        previous_price: Previous price
        
    Returns:
        Dictionary with OI analysis
    """
    result = {
        "oi_buildup_signal": 0,
        "oi_change_pct": np.nan,
        "oi_pattern": "unknown",
    }
    
    # Calculate total OI change
    curr_total = current_oi.get("call", 0) + current_oi.get("put", 0)
    prev_total = previous_oi.get("call", 0) + previous_oi.get("put", 0)
    
    if prev_total == 0:
        return result
    
    oi_change = (curr_total - prev_total) / prev_total
    price_change = (current_price - previous_price) / previous_price if previous_price > 0 else 0
    
    result["oi_change_pct"] = oi_change
    
    # Classify pattern
    oi_up = oi_change > 0.01  # OI increased > 1%
    oi_down = oi_change < -0.01  # OI decreased > 1%
    price_up = price_change > 0.005  # Price up > 0.5%
    price_down = price_change < -0.005  # Price down > 0.5%
    
    if oi_up and price_up:
        result["oi_pattern"] = "long_buildup"
        result["oi_buildup_signal"] = 1  # Bullish
    elif oi_up and price_down:
        result["oi_pattern"] = "short_buildup"
        result["oi_buildup_signal"] = -1  # Bearish
    elif oi_down and price_down:
        result["oi_pattern"] = "long_unwinding"
        result["oi_buildup_signal"] = -0.5  # Weakly bearish
    elif oi_down and price_up:
        result["oi_pattern"] = "short_covering"
        result["oi_buildup_signal"] = 0.5  # Weakly bullish
    else:
        result["oi_pattern"] = "neutral"
        result["oi_buildup_signal"] = 0
    
    return result


def add_options_features_to_df(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    close_col: str = "close",
    nifty_pcr: Optional[float] = None,
    config: Optional[OptionsFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Add options features to a DataFrame.
    
    Args:
        df: Input DataFrame with ticker and price data
        ticker_col: Column name for ticker
        close_col: Column name for close price
        nifty_pcr: Current NIFTY PCR value
        config: Feature configuration
        
    Returns:
        DataFrame with added options features
    """
    df = df.copy()
    config = config or OptionsFeatureConfig()
    
    # Add NIFTY PCR features (same for all stocks)
    if nifty_pcr is not None:
        df["nifty_pcr"] = nifty_pcr
        df["nifty_pcr_zscore"] = _compute_pcr_zscore(nifty_pcr, None, config)
        df["nifty_pcr_sentiment"] = _classify_pcr_sentiment(nifty_pcr, config)
    else:
        df["nifty_pcr"] = np.nan
        df["nifty_pcr_zscore"] = np.nan
        df["nifty_pcr_sentiment"] = 0
    
    # Add F&O flag
    df["is_fno_stock"] = df[ticker_col].apply(is_fno_stock).astype(float)
    
    # Add placeholders for stock-level features (require live data)
    df["stock_pcr"] = np.nan
    df["stock_pcr_zscore"] = np.nan
    df["max_pain_distance"] = np.nan
    
    return df


def get_pcr_signal(pcr: float, threshold_high: float = 1.2, threshold_low: float = 0.7) -> str:
    """
    Get simple PCR signal interpretation.
    
    Args:
        pcr: Put-Call Ratio value
        threshold_high: High PCR threshold (contrarian bullish)
        threshold_low: Low PCR threshold (contrarian bearish)
        
    Returns:
        Signal string: "bullish", "bearish", or "neutral"
    """
    if pcr > threshold_high:
        return "bullish"  # Contrarian: excessive puts = market fear = buy
    elif pcr < threshold_low:
        return "bearish"  # Contrarian: excessive calls = market greed = sell
    return "neutral"
