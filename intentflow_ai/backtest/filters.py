"""Reusable risk and regime filters for the backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskFilterConfig:
    """Price-based regime controls."""

    trend_fast: int = 50
    trend_slow: int = 200
    vol_lookback: int = 20
    vol_high: float = 0.04
    allow_high_vol: bool = False
    allow_downtrend: bool = False
    max_positions: int | None = None
    cooldown_days: int = 0


@dataclass
class MetaFilterConfig:
    """Meta-model gating config."""

    enabled: bool = False
    proba_col: str = "meta_proba"
    min_prob: float = 0.5


@dataclass
class ConcentrationConfig:
    """Portfolio concentration limits."""
    
    # Maximum allocation per sector (as fraction)
    max_sector_exposure: float = 0.30  # 30%
    
    # Maximum allocation per single stock (as fraction)
    max_single_stock: float = 0.10  # 10%
    
    # Minimum number of sectors for diversification
    min_sectors: int = 4
    
    # Whether to rebalance to limits or hard cap
    rebalance_to_limits: bool = True


class SectorConcentrationFilter:
    """
    Apply sector concentration limits to signal selection.
    
    Ensures portfolio diversification by:
    1. Limiting exposure to any single sector
    2. Limiting exposure to any single stock
    3. Ensuring minimum sector diversification
    
    Usage:
        filter = SectorConcentrationFilter(ConcentrationConfig())
        filtered_picks = filter.apply(picks, sector_map)
    """
    
    def __init__(self, config: Optional[ConcentrationConfig] = None):
        self.config = config or ConcentrationConfig()
    
    def apply(
        self,
        picks: List[Dict],
        sector_map: Dict[str, str],
        weights: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Apply concentration limits to picks.
        
        Args:
            picks: List of pick dictionaries with 'ticker' key
            sector_map: Mapping of ticker -> sector
            weights: Optional list of weights (equal weight if None)
            
        Returns:
            Filtered and potentially reweighted picks
        """
        if not picks:
            return picks
        
        # Assign sectors
        for pick in picks:
            ticker = pick.get("ticker", "")
            pick["sector"] = sector_map.get(ticker, "Unknown")
        
        # Equal weights if not provided
        n_picks = len(picks)
        if weights is None:
            weights = [1.0 / n_picks] * n_picks
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Apply single stock limit
        weights = self._apply_single_stock_limit(weights)
        
        # Apply sector limits
        picks, weights = self._apply_sector_limits(picks, weights)
        
        # Ensure minimum diversification
        picks = self._ensure_min_diversification(picks)
        
        # Update weights in picks
        for i, pick in enumerate(picks):
            if i < len(weights):
                pick["weight"] = weights[i]
        
        return picks
    
    def _apply_single_stock_limit(self, weights: List[float]) -> List[float]:
        """Cap individual stock weights."""
        max_weight = self.config.max_single_stock
        
        # Iteratively redistribute excess weight
        for _ in range(10):  # Max iterations
            excess = 0.0
            uncapped_count = 0
            
            for i, w in enumerate(weights):
                if w > max_weight:
                    excess += w - max_weight
                    weights[i] = max_weight
                elif w < max_weight:
                    uncapped_count += 1
            
            if excess == 0 or uncapped_count == 0:
                break
            
            # Redistribute excess to uncapped
            redistribution = excess / uncapped_count
            for i in range(len(weights)):
                if weights[i] < max_weight:
                    weights[i] += redistribution
        
        # Renormalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    def _apply_sector_limits(
        self,
        picks: List[Dict],
        weights: List[float],
    ) -> Tuple[List[Dict], List[float]]:
        """Apply sector concentration limits."""
        max_sector = self.config.max_sector_exposure
        
        # Calculate sector weights
        sector_weights: Dict[str, float] = {}
        for pick, weight in zip(picks, weights):
            sector = pick.get("sector", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        # Find over-concentrated sectors
        over_sectors = {s: w for s, w in sector_weights.items() if w > max_sector}
        
        if not over_sectors:
            return picks, weights
        
        logger.info(
            "Applying sector limits",
            extra={"over_concentrated": over_sectors}
        )
        
        # Scale down weights for over-concentrated sectors
        new_weights = []
        for pick, weight in zip(picks, weights):
            sector = pick.get("sector", "Unknown")
            if sector in over_sectors:
                # Scale factor to bring sector to limit
                scale = max_sector / over_sectors[sector]
                new_weights.append(weight * scale)
            else:
                new_weights.append(weight)
        
        # Renormalize
        total = sum(new_weights)
        if total > 0:
            new_weights = [w / total for w in new_weights]
        
        return picks, new_weights
    
    def _ensure_min_diversification(self, picks: List[Dict]) -> List[Dict]:
        """Ensure minimum sector diversification."""
        min_sectors = self.config.min_sectors
        
        # Count unique sectors
        sectors = set(pick.get("sector", "Unknown") for pick in picks)
        
        if len(sectors) >= min_sectors:
            return picks
        
        logger.warning(
            f"Only {len(sectors)} sectors in picks, minimum is {min_sectors}",
            extra={"sectors": list(sectors)}
        )
        
        # Can't add more sectors here, but flag the issue
        return picks
    
    def get_sector_distribution(
        self,
        picks: List[Dict],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get sector weight distribution."""
        if weights is None:
            weights = [1.0 / len(picks)] * len(picks) if picks else []
        
        sector_weights: Dict[str, float] = {}
        for pick, weight in zip(picks, weights):
            sector = pick.get("sector", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return sector_weights


def apply_sector_limits(
    picks: List[Dict],
    sector_map: Dict[str, str],
    max_per_sector: float = 0.30,
) -> List[Dict]:
    """
    Convenience function to apply sector concentration limits.
    
    Args:
        picks: List of pick dictionaries with 'ticker' key
        sector_map: Mapping of ticker -> sector
        max_per_sector: Maximum allocation per sector (default 30%)
        
    Returns:
        Filtered picks with weights
    """
    config = ConcentrationConfig(max_sector_exposure=max_per_sector)
    filter = SectorConcentrationFilter(config)
    return filter.apply(picks, sector_map)


def compute_regime_flags(px: pd.DataFrame, cfg: RiskFilterConfig) -> pd.DataFrame:
    """Return per-date flags for trend/volatility regimes."""

    idx = px.mean(axis=1)
    ma_fast = idx.rolling(cfg.trend_fast).mean()
    ma_slow = idx.rolling(cfg.trend_slow).mean()
    trend_ok = ma_fast > ma_slow

    ret = idx.pct_change()
    vol = ret.rolling(cfg.vol_lookback).std().fillna(0)
    vol_ok = vol < cfg.vol_high if cfg.vol_high > 0 else pd.Series(True, index=idx.index)

    allow = trend_ok | cfg.allow_downtrend
    if not cfg.allow_high_vol:
        allow = allow & vol_ok

    return pd.DataFrame(
        {
            "trend_ok": trend_ok,
            "vol_ok": vol_ok,
            "allow_entry": allow,
            "index_vol": vol,
        }
    )


def apply_cooldown(tickers: Iterable[str], cooldown_state: Dict[str, pd.Timestamp], current: pd.Timestamp) -> List[str]:
    """Filter tickers that are still in cooldown."""

    allowed = []
    for t in tickers:
        until = cooldown_state.get(t)
        if until is None or until <= current:
            allowed.append(t)
    return allowed


def update_cooldown(cooldown_state: Dict[str, pd.Timestamp], tickers: Iterable[str], current: pd.Timestamp, days: int) -> None:
    """Mark tickers as on cooldown until `current + days`."""

    if days <= 0:
        return
    for t in tickers:
        cooldown_state[t] = current + pd.Timedelta(days=days)
