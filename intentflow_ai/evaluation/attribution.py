"""
Performance Attribution Module

Breaks down strategy returns into component contributions:
1. Factor attribution (momentum, value, quality, etc.)
2. Sector attribution (over/underweight effects)
3. Stock selection (alpha from individual picks)
4. Timing attribution (entry/exit timing effects)

Essential for understanding where alpha comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AttributionConfig:
    """Configuration for performance attribution."""
    
    # Factor definitions
    momentum_lookback: int = 20
    value_features: List[str] = field(default_factory=lambda: [
        "pe_ratio", "pb_ratio", "ev_ebitda"
    ])
    quality_features: List[str] = field(default_factory=lambda: [
        "roe", "roa", "gross_margin", "f_score"
    ])
    
    # Benchmark
    benchmark_ticker: str = "^NSEI"  # NIFTY 50
    risk_free_rate: float = 0.06    # 6% annual
    
    # Analysis periods
    attribution_window_days: int = 15  # Match rebalancing


@dataclass
class FactorReturn:
    """Factor return attribution."""
    
    factor_name: str
    factor_return: float
    portfolio_exposure: float
    contribution: float  # exposure * return
    t_stat: Optional[float] = None


@dataclass
class SectorAttribution:
    """Sector-level attribution."""
    
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    active_weight: float  # portfolio - benchmark
    sector_return: float
    allocation_effect: float   # Active weight * benchmark return
    selection_effect: float    # Portfolio weight * (port return - bench return)
    total_effect: float


@dataclass 
class AttributionReport:
    """Complete attribution report."""
    
    period_start: str
    period_end: str
    total_return: float
    benchmark_return: float
    active_return: float  # alpha
    
    # Factor attribution
    factor_contributions: List[FactorReturn]
    factor_explained_return: float
    
    # Sector attribution
    sector_contributions: List[SectorAttribution]
    allocation_effect: float
    selection_effect: float
    
    # Risk metrics
    tracking_error: float
    information_ratio: float
    
    # Top contributors
    top_contributors: List[Dict[str, Any]]
    top_detractors: List[Dict[str, Any]]


class PerformanceAttributor:
    """
    Decompose strategy returns into factor and sector contributions.
    
    Usage:
        attributor = PerformanceAttributor()
        
        report = attributor.attribute(
            portfolio_returns=my_returns,
            portfolio_holdings=my_holdings,
            benchmark_returns=nifty_returns,
            features_df=features,
        )
        
        print(f"Alpha: {report.active_return:.2%}")
        print(f"Factor contribution: {report.factor_explained_return:.2%}")
    """
    
    def __init__(self, config: Optional[AttributionConfig] = None):
        self.config = config or AttributionConfig()
    
    def attribute(
        self,
        portfolio_returns: pd.Series,
        portfolio_holdings: pd.DataFrame,
        benchmark_returns: pd.Series,
        features_df: Optional[pd.DataFrame] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> AttributionReport:
        """
        Run full performance attribution.
        
        Args:
            portfolio_returns: Daily portfolio returns
            portfolio_holdings: Holdings with [date, ticker, weight, return]
            benchmark_returns: Benchmark daily returns
            features_df: Feature data for factor attribution
            sector_map: Ticker to sector mapping
            
        Returns:
            AttributionReport with full breakdown
        """
        # Align time series
        portfolio_returns = portfolio_returns.dropna()
        benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
        
        period_start = str(portfolio_returns.index.min().date())
        period_end = str(portfolio_returns.index.max().date())
        
        # Calculate total returns
        total_return = (1 + portfolio_returns).prod() - 1
        benchmark_return = (1 + benchmark_returns).prod() - 1
        active_return = total_return - benchmark_return
        
        # Factor attribution
        if features_df is not None:
            factor_contributions = self._factor_attribution(
                portfolio_holdings, features_df
            )
            factor_explained = sum(f.contribution for f in factor_contributions)
        else:
            factor_contributions = []
            factor_explained = 0.0
        
        # Sector attribution
        if sector_map and not portfolio_holdings.empty:
            sector_contributions = self._sector_attribution(
                portfolio_holdings, benchmark_returns, sector_map
            )
            allocation_effect = sum(s.allocation_effect for s in sector_contributions)
            selection_effect = sum(s.selection_effect for s in sector_contributions)
        else:
            sector_contributions = []
            allocation_effect = 0.0
            selection_effect = 0.0
        
        # Risk metrics
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        if tracking_error > 0:
            information_ratio = (active_return * 252 / len(portfolio_returns)) / tracking_error
        else:
            information_ratio = 0.0
        
        # Top contributors/detractors
        top_contributors, top_detractors = self._get_top_stocks(portfolio_holdings)
        
        return AttributionReport(
            period_start=period_start,
            period_end=period_end,
            total_return=total_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            factor_contributions=factor_contributions,
            factor_explained_return=factor_explained,
            sector_contributions=sector_contributions,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            top_contributors=top_contributors,
            top_detractors=top_detractors,
        )
    
    def _factor_attribution(
        self,
        holdings: pd.DataFrame,
        features: pd.DataFrame,
    ) -> List[FactorReturn]:
        """Attribute returns to style factors."""
        factor_returns = []
        
        if holdings.empty or features.empty:
            return factor_returns
        
        # Define factor groups
        factors = {
            "momentum": ["price_ret_20d", "price_mom_20", "momentum_ratio_10_30"],
            "value": ["pe_ratio_inv", "pb_ratio_inv", "ev_ebitda_inv"],
            "quality": ["roe", "roa", "gross_margin"],
            "volatility": ["price_vol_20", "vol_ratio_short_long"],
            "size": ["log_market_cap", "mcap_percentile"],
        }
        
        for factor_name, factor_cols in factors.items():
            # Get available factor columns
            available_cols = [c for c in factor_cols if c in features.columns]
            if not available_cols:
                continue
            
            # Calculate factor exposure (average across holdings)
            try:
                factor_exposure = self._calculate_factor_exposure(
                    holdings, features, available_cols
                )
                
                # Estimate factor return from cross-sectional regression
                factor_return = self._estimate_factor_return(
                    features, available_cols, holdings
                )
                
                contribution = factor_exposure * factor_return
                
                factor_returns.append(FactorReturn(
                    factor_name=factor_name,
                    factor_return=factor_return,
                    portfolio_exposure=factor_exposure,
                    contribution=contribution,
                ))
            except Exception as e:
                logger.debug(f"Could not compute {factor_name} attribution: {e}")
        
        return factor_returns
    
    def _calculate_factor_exposure(
        self,
        holdings: pd.DataFrame,
        features: pd.DataFrame,
        factor_cols: List[str],
    ) -> float:
        """Calculate portfolio's exposure to a factor."""
        if "ticker" not in holdings.columns or "ticker" not in features.columns:
            return 0.0
        
        # Merge holdings with features
        merged = holdings.merge(
            features[["ticker"] + factor_cols].drop_duplicates("ticker"),
            on="ticker",
            how="left"
        )
        
        if merged.empty:
            return 0.0
        
        # Weighted average exposure
        weight_col = "weight" if "weight" in merged.columns else None
        if weight_col and merged[weight_col].sum() > 0:
            weights = merged[weight_col] / merged[weight_col].sum()
        else:
            weights = pd.Series(1 / len(merged), index=merged.index)
        
        # Average factor exposure (standardized)
        factor_values = merged[factor_cols].mean(axis=1)
        exposure = (factor_values * weights).sum()
        
        return exposure
    
    def _estimate_factor_return(
        self,
        features: pd.DataFrame,
        factor_cols: List[str],
        holdings: pd.DataFrame,
    ) -> float:
        """Estimate factor return from cross-sectional data."""
        if "return" not in features.columns:
            return 0.0
        
        # Simple correlation-based estimate
        factor_composite = features[factor_cols].mean(axis=1)
        returns = features["return"]
        
        valid_idx = factor_composite.notna() & returns.notna()
        if valid_idx.sum() < 10:
            return 0.0
        
        correlation = factor_composite[valid_idx].corr(returns[valid_idx])
        
        # Scale to approximate return
        return correlation * returns[valid_idx].std()
    
    def _sector_attribution(
        self,
        holdings: pd.DataFrame,
        benchmark_returns: pd.Series,
        sector_map: Dict[str, str],
    ) -> List[SectorAttribution]:
        """Attribute returns to sector allocation and selection."""
        attributions = []
        
        if "ticker" not in holdings.columns:
            return attributions
        
        # Add sector to holdings
        holdings = holdings.copy()
        holdings["sector"] = holdings["ticker"].map(sector_map).fillna("Unknown")
        
        # Calculate sector weights in portfolio
        weight_col = "weight" if "weight" in holdings.columns else None
        
        if weight_col:
            portfolio_weights = holdings.groupby("sector")[weight_col].sum()
            portfolio_weights = portfolio_weights / portfolio_weights.sum()
        else:
            portfolio_weights = holdings.groupby("sector").size()
            portfolio_weights = portfolio_weights / portfolio_weights.sum()
        
        # Assume equal sector weights in benchmark (simplified)
        all_sectors = list(set(sector_map.values()))
        benchmark_weights = pd.Series(
            1 / len(all_sectors), index=all_sectors
        )
        
        # Calculate returns by sector
        return_col = "return" if "return" in holdings.columns else None
        if return_col:
            sector_returns = holdings.groupby("sector")[return_col].mean()
        else:
            sector_returns = pd.Series(0.0, index=portfolio_weights.index)
        
        # Benchmark sector return (use overall benchmark as proxy)
        bench_return = benchmark_returns.mean()
        
        for sector in portfolio_weights.index:
            port_weight = portfolio_weights.get(sector, 0)
            bench_weight = benchmark_weights.get(sector, 1 / len(all_sectors))
            active_weight = port_weight - bench_weight
            
            sector_return = sector_returns.get(sector, 0)
            
            # Brinson attribution
            allocation = active_weight * bench_return
            selection = port_weight * (sector_return - bench_return)
            
            attributions.append(SectorAttribution(
                sector=sector,
                portfolio_weight=port_weight,
                benchmark_weight=bench_weight,
                active_weight=active_weight,
                sector_return=sector_return,
                allocation_effect=allocation,
                selection_effect=selection,
                total_effect=allocation + selection,
            ))
        
        return attributions
    
    def _get_top_stocks(
        self,
        holdings: pd.DataFrame,
        n: int = 5,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Get top contributing and detracting stocks."""
        if holdings.empty or "return" not in holdings.columns:
            return [], []
        
        weight_col = "weight" if "weight" in holdings.columns else None
        
        if weight_col:
            holdings["contribution"] = holdings[weight_col] * holdings["return"]
        else:
            holdings["contribution"] = holdings["return"] / len(holdings)
        
        # Sort by contribution
        sorted_df = holdings.sort_values("contribution", ascending=False)
        
        top = sorted_df.head(n)[["ticker", "return", "contribution"]].to_dict("records")
        bottom = sorted_df.tail(n)[["ticker", "return", "contribution"]].to_dict("records")
        
        return top, bottom


def generate_attribution_summary(report: AttributionReport) -> str:
    """Generate human-readable attribution summary."""
    lines = [
        "=" * 60,
        "PERFORMANCE ATTRIBUTION REPORT",
        "=" * 60,
        f"Period: {report.period_start} to {report.period_end}",
        "",
        "--- RETURNS ---",
        f"Portfolio Return:  {report.total_return:+.2%}",
        f"Benchmark Return:  {report.benchmark_return:+.2%}",
        f"Active Return:     {report.active_return:+.2%}",
        "",
        "--- RISK METRICS ---",
        f"Tracking Error:    {report.tracking_error:.2%}",
        f"Information Ratio: {report.information_ratio:.2f}",
        "",
    ]
    
    if report.factor_contributions:
        lines.append("--- FACTOR ATTRIBUTION ---")
        for fc in report.factor_contributions:
            lines.append(
                f"  {fc.factor_name:12s}: {fc.contribution:+.2%} "
                f"(exposure: {fc.portfolio_exposure:.2f})"
            )
        lines.append(f"  {'Total':12s}: {report.factor_explained_return:+.2%}")
        lines.append("")
    
    if report.sector_contributions:
        lines.append("--- SECTOR ATTRIBUTION ---")
        for sc in sorted(report.sector_contributions, key=lambda x: -abs(x.total_effect))[:5]:
            lines.append(
                f"  {sc.sector:20s}: {sc.total_effect:+.2%} "
                f"(weight: {sc.active_weight:+.1%})"
            )
        lines.append(f"  Allocation Effect:  {report.allocation_effect:+.2%}")
        lines.append(f"  Selection Effect:   {report.selection_effect:+.2%}")
        lines.append("")
    
    if report.top_contributors:
        lines.append("--- TOP CONTRIBUTORS ---")
        for stock in report.top_contributors:
            lines.append(f"  {stock['ticker']:12s}: {stock['contribution']:+.2%}")
        lines.append("")
    
    if report.top_detractors:
        lines.append("--- TOP DETRACTORS ---")
        for stock in report.top_detractors:
            lines.append(f"  {stock['ticker']:12s}: {stock['contribution']:+.2%}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
