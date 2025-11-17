"""Market regime classification utilities for production trading.

Provides robust regime detection using multiple signals:
- Volatility regimes (low/medium/high)
- Trend regimes (uptrend/downtrend/sideways)
- Market breadth (strong/weak)
- VIX-like indicators when available

These regimes help filter trades and segment model performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class VolatilityRegime(Enum):
    """Volatility-based regime categories."""
    LOW = "low_vol"
    MEDIUM = "medium_vol"
    HIGH = "high_vol"
    EXTREME = "extreme_vol"


class TrendRegime(Enum):
    """Trend-based regime categories."""
    STRONG_UP = "strong_uptrend"
    WEAK_UP = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_downtrend"


class BreadthRegime(Enum):
    """Market breadth categories."""
    STRONG = "strong_breadth"
    NEUTRAL = "neutral_breadth"
    WEAK = "weak_breadth"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    
    # Volatility thresholds (annualized)
    vol_low_threshold: float = 0.15
    vol_medium_threshold: float = 0.25
    vol_high_threshold: float = 0.40
    vol_lookback: int = 20
    
    # Trend parameters
    trend_fast_ma: int = 50
    trend_slow_ma: int = 200
    trend_strength_threshold: float = 0.02  # 2% above/below MAs
    
    # Breadth parameters
    breadth_lookback: int = 20
    breadth_strong_threshold: float = 0.60  # 60% stocks above MA
    breadth_weak_threshold: float = 0.40
    
    # Risk-off detection
    drawdown_threshold: float = -0.10  # 10% from peak
    use_vix_proxy: bool = True


@dataclass
class RegimeClassifier:
    """Production-grade multi-dimensional regime classifier.
    
    Combines volatility, trend, and breadth signals to identify market regimes.
    Used to filter trades (avoid high-vol, downtrend periods) and segment model performance.
    """

    cfg: RegimeConfig = field(default_factory=RegimeConfig)
    
    def infer(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes from price panel.
        
        Args:
            price_panel: DataFrame with columns [date, ticker, close]
            
        Returns:
            DataFrame indexed by date with regime columns:
            - volatility_regime
            - trend_regime  
            - breadth_regime
            - composite_regime
            - allow_entry (bool)
            - regime_score (0-100, higher = more favorable)
        """
        if price_panel.empty:
            return pd.DataFrame()
            
        # Compute market-level aggregates
        market_prices = price_panel.groupby("date")["close"].mean().sort_index()
        
        if len(market_prices) < max(self.cfg.trend_slow_ma, self.cfg.vol_lookback):
            logger.warning("Insufficient data for regime detection")
            return self._default_regimes(market_prices.index)
        
        # Volatility regime
        vol_regime = self._detect_volatility_regime(market_prices)
        
        # Trend regime
        trend_regime = self._detect_trend_regime(market_prices)
        
        # Breadth regime
        breadth_regime = self._detect_breadth_regime(price_panel)
        
        # Drawdown detection
        drawdown = self._compute_drawdown(market_prices)
        
        # Combine into composite regime
        regime_df = pd.DataFrame({
            "volatility_regime": vol_regime,
            "trend_regime": trend_regime,
            "breadth_regime": breadth_regime,
            "drawdown": drawdown,
        }, index=market_prices.index)
        
        # Composite regime and entry filter
        regime_df["composite_regime"] = self._create_composite_regime(regime_df)
        regime_df["allow_entry"] = self._should_allow_entry(regime_df)
        regime_df["regime_score"] = self._compute_regime_score(regime_df)
        
        return regime_df
    
    def _detect_volatility_regime(self, prices: pd.Series) -> pd.Series:
        """Classify volatility regime."""
        returns = prices.pct_change()
        realized_vol = returns.rolling(self.cfg.vol_lookback).std() * np.sqrt(252)
        
        def classify_vol(vol):
            if pd.isna(vol):
                return VolatilityRegime.MEDIUM.value
            elif vol < self.cfg.vol_low_threshold:
                return VolatilityRegime.LOW.value
            elif vol < self.cfg.vol_medium_threshold:
                return VolatilityRegime.MEDIUM.value
            elif vol < self.cfg.vol_high_threshold:
                return VolatilityRegime.HIGH.value
            else:
                return VolatilityRegime.EXTREME.value
        
        return realized_vol.apply(classify_vol)
    
    def _detect_trend_regime(self, prices: pd.Series) -> pd.Series:
        """Classify trend regime using moving averages."""
        fast_ma = prices.rolling(self.cfg.trend_fast_ma, min_periods=1).mean()
        slow_ma = prices.rolling(self.cfg.trend_slow_ma, min_periods=1).mean()
        
        # Price relative to MAs
        price_vs_fast = (prices / fast_ma) - 1.0
        price_vs_slow = (prices / slow_ma) - 1.0
        ma_spread = (fast_ma / slow_ma) - 1.0
        
        def classify_trend(row):
            p_fast, p_slow, spread = row
            thresh = self.cfg.trend_strength_threshold
            
            if pd.isna(p_fast) or pd.isna(p_slow):
                return TrendRegime.SIDEWAYS.value
            
            # Strong uptrend: price > both MAs, fast > slow
            if p_fast > thresh and p_slow > thresh and spread > thresh:
                return TrendRegime.STRONG_UP.value
            # Weak uptrend
            elif p_fast > 0 and p_slow > 0:
                return TrendRegime.WEAK_UP.value
            # Strong downtrend
            elif p_fast < -thresh and p_slow < -thresh and spread < -thresh:
                return TrendRegime.STRONG_DOWN.value
            # Weak downtrend
            elif p_fast < 0 and p_slow < 0:
                return TrendRegime.WEAK_DOWN.value
            else:
                return TrendRegime.SIDEWAYS.value
        
        trend_df = pd.DataFrame({
            "p_fast": price_vs_fast,
            "p_slow": price_vs_slow,
            "spread": ma_spread
        })
        
        return trend_df.apply(classify_trend, axis=1)
    
    def _detect_breadth_regime(self, price_panel: pd.DataFrame) -> pd.Series:
        """Classify breadth regime based on % of stocks above MA."""
        if "ticker" not in price_panel.columns:
            dates = price_panel.groupby("date")["close"].mean().index
            return pd.Series(BreadthRegime.NEUTRAL.value, index=dates)
        
        # For each ticker, check if price > MA
        price_panel = price_panel.sort_values(["ticker", "date"]).copy()
        price_panel["ma_20"] = price_panel.groupby("ticker")["close"].transform(
            lambda x: x.rolling(self.cfg.breadth_lookback, min_periods=1).mean()
        )
        price_panel["above_ma"] = (price_panel["close"] > price_panel["ma_20"]).astype(int)
        
        # Daily breadth: % of tickers above their MA
        breadth = price_panel.groupby("date")["above_ma"].mean()
        
        def classify_breadth(pct):
            if pd.isna(pct):
                return BreadthRegime.NEUTRAL.value
            elif pct >= self.cfg.breadth_strong_threshold:
                return BreadthRegime.STRONG.value
            elif pct <= self.cfg.breadth_weak_threshold:
                return BreadthRegime.WEAK.value
            else:
                return BreadthRegime.NEUTRAL.value
        
        return breadth.apply(classify_breadth)
    
    def _compute_drawdown(self, prices: pd.Series) -> pd.Series:
        """Compute drawdown from running maximum."""
        running_max = prices.expanding().max()
        drawdown = (prices / running_max) - 1.0
        return drawdown.fillna(0.0)
    
    def _create_composite_regime(self, regime_df: pd.DataFrame) -> pd.Series:
        """Create human-readable composite regime label."""
        def composite(row):
            vol = row["volatility_regime"]
            trend = row["trend_regime"]
            breadth = row["breadth_regime"]
            
            # Combine into descriptive label
            if "high" in vol or "extreme" in vol:
                return f"high_vol_{trend}"
            elif "strong_up" in trend and "strong" in breadth:
                return "bull_market"
            elif "strong_down" in trend or "weak" in breadth:
                return "bear_market"
            else:
                return f"{vol}_{trend}"
        
        return regime_df.apply(composite, axis=1)
    
    def _should_allow_entry(self, regime_df: pd.DataFrame) -> pd.Series:
        """Determine whether to allow new entries in each regime.
        
        Conservative filter: block trades in:
        - High/extreme volatility
        - Strong downtrends
        - Large drawdowns
        - Weak breadth
        """
        conditions = [
            ~regime_df["volatility_regime"].isin([
                VolatilityRegime.HIGH.value,
                VolatilityRegime.EXTREME.value
            ]),
            ~regime_df["trend_regime"].isin([
                TrendRegime.STRONG_DOWN.value
            ]),
            regime_df["drawdown"] > self.cfg.drawdown_threshold,
            regime_df["breadth_regime"] != BreadthRegime.WEAK.value
        ]
        
        # All conditions must be true
        allow = pd.Series(True, index=regime_df.index)
        for cond in conditions:
            allow &= cond
        
        return allow
    
    def _compute_regime_score(self, regime_df: pd.DataFrame) -> pd.Series:
        """Compute 0-100 regime favorability score.
        
        Higher score = more favorable for trading.
        """
        score = pd.Series(50.0, index=regime_df.index)  # Start neutral
        
        # Volatility scoring
        vol_map = {
            VolatilityRegime.LOW.value: 20,
            VolatilityRegime.MEDIUM.value: 10,
            VolatilityRegime.HIGH.value: -15,
            VolatilityRegime.EXTREME.value: -30
        }
        for regime, delta in vol_map.items():
            score[regime_df["volatility_regime"] == regime] += delta
        
        # Trend scoring
        trend_map = {
            TrendRegime.STRONG_UP.value: 25,
            TrendRegime.WEAK_UP.value: 10,
            TrendRegime.SIDEWAYS.value: 0,
            TrendRegime.WEAK_DOWN.value: -10,
            TrendRegime.STRONG_DOWN.value: -25
        }
        for regime, delta in trend_map.items():
            score[regime_df["trend_regime"] == regime] += delta
        
        # Breadth scoring
        breadth_map = {
            BreadthRegime.STRONG.value: 15,
            BreadthRegime.NEUTRAL.value: 0,
            BreadthRegime.WEAK.value: -15
        }
        for regime, delta in breadth_map.items():
            score[regime_df["breadth_regime"] == regime] += delta
        
        # Drawdown penalty
        score -= regime_df["drawdown"].abs() * 100  # -10 points per 10% drawdown
        
        return score.clip(0, 100)
    
    def _default_regimes(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Return default neutral regimes when insufficient data."""
        return pd.DataFrame({
            "volatility_regime": VolatilityRegime.MEDIUM.value,
            "trend_regime": TrendRegime.SIDEWAYS.value,
            "breadth_regime": BreadthRegime.NEUTRAL.value,
            "composite_regime": "neutral",
            "allow_entry": True,
            "regime_score": 50.0,
            "drawdown": 0.0,
        }, index=dates)
    
    def get_regime_summary(self, regime_df: pd.DataFrame) -> Dict[str, float]:
        """Generate summary statistics of regime distribution.
        
        Useful for monitoring and reporting.
        """
        if regime_df.empty:
            return {}
        
        total_days = len(regime_df)
        
        summary = {
            "total_days": total_days,
            "allow_entry_pct": (regime_df["allow_entry"].sum() / total_days * 100),
            "avg_regime_score": regime_df["regime_score"].mean(),
            "max_drawdown": regime_df["drawdown"].min(),
        }
        
        # Regime distributions
        for col in ["volatility_regime", "trend_regime", "breadth_regime"]:
            dist = regime_df[col].value_counts()
            for regime, count in dist.items():
                summary[f"{col}_{regime}_pct"] = (count / total_days * 100)
        
        return summary


def apply_regime_filter_to_signals(
    signals: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    date_col: str = "date",
    require_entry_allowed: bool = True,
    min_regime_score: float = 30.0,
) -> pd.DataFrame:
    """Filter signals based on regime conditions.
    
    Args:
        signals: DataFrame with trading signals
        regime_df: Regime classifications from RegimeClassifier
        require_entry_allowed: If True, drop signals where allow_entry=False
        min_regime_score: Minimum regime score to keep signal
        
    Returns:
        Filtered signals DataFrame
    """
    if signals.empty or regime_df.empty:
        return signals
    
    # Merge regime data
    signals = signals.copy()
    signals[date_col] = pd.to_datetime(signals[date_col])
    
    merged = signals.merge(
        regime_df[["allow_entry", "regime_score", "composite_regime"]],
        left_on=date_col,
        right_index=True,
        how="left"
    )
    
    original_count = len(merged)
    
    # Apply filters
    if require_entry_allowed:
        merged = merged[merged["allow_entry"].fillna(False)]
    
    merged = merged[merged["regime_score"].fillna(0) >= min_regime_score]
    
    filtered_count = len(merged)
    if filtered_count < original_count:
        logger.info(
            "Regime filter applied",
            extra={
                "original_signals": original_count,
                "filtered_signals": filtered_count,
                "removed_pct": (1 - filtered_count/original_count) * 100
            }
        )
    
    return merged
