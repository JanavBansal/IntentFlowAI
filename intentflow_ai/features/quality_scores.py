"""
Quality Score Features

Implements classical quality metrics:
1. Piotroski F-Score (9-point score for financial strength)
2. Altman Z-Score (bankruptcy predictor)
3. Beneish M-Score (earnings manipulation detector)
4. Quality composite score

These are proven fundamental factors with strong predictive power.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def piotroski_f_score(fundamentals: Dict[str, Any]) -> int:
    """
    Calculate Piotroski F-Score (0-9).
    
    The F-Score is a discrete score that ranges from 0-9 and captures
    a firm's financial position across three dimensions:
    - Profitability (4 signals)
    - Leverage/Liquidity (3 signals)
    - Operating Efficiency (2 signals)
    
    Higher scores indicate stronger financial position.
    Stocks with F-Score >= 7 are considered strong.
    Stocks with F-Score <= 3 are considered weak.
    
    Args:
        fundamentals: Dictionary with fundamental data including:
            - net_income, operating_cashflow, roa, roa_prev
            - total_assets, total_debt, current_ratio
            - shares_outstanding, gross_margin, asset_turnover
            
    Returns:
        F-Score (0-9)
    """
    score = 0
    
    # === PROFITABILITY (4 signals) ===
    
    # 1. ROA > 0 (profitable)
    roa = fundamentals.get("roa")
    if roa is not None and roa > 0:
        score += 1
    
    # 2. Operating Cash Flow > 0
    ocf = fundamentals.get("operating_cashflow")
    if ocf is not None and ocf > 0:
        score += 1
    
    # 3. ROA improving (current > previous)
    roa_prev = fundamentals.get("roa_prev") or fundamentals.get("roa_1y_ago")
    if roa is not None and roa_prev is not None and roa > roa_prev:
        score += 1
    
    # 4. Cash flow quality (OCF > Net Income, accruals signal)
    net_income = fundamentals.get("net_income")
    if ocf is not None and net_income is not None:
        if ocf > net_income:
            score += 1
    
    # === LEVERAGE, LIQUIDITY, SOURCE OF FUNDS (3 signals) ===
    
    # 5. Decreasing leverage (Long-term debt / Total assets)
    total_debt = fundamentals.get("total_debt", 0) or 0
    total_assets = fundamentals.get("total_assets", 1) or 1
    debt_prev = fundamentals.get("total_debt_prev") or fundamentals.get("total_debt_1y_ago")
    assets_prev = fundamentals.get("total_assets_prev") or fundamentals.get("total_assets_1y_ago")
    
    leverage_curr = total_debt / total_assets if total_assets > 0 else 0
    if debt_prev is not None and assets_prev is not None and assets_prev > 0:
        leverage_prev = debt_prev / assets_prev
        if leverage_curr < leverage_prev:
            score += 1
    elif leverage_curr < 0.5:  # Low leverage is good
        score += 1
    
    # 6. Improving current ratio
    current_ratio = fundamentals.get("current_ratio")
    current_ratio_prev = fundamentals.get("current_ratio_prev") or fundamentals.get("current_ratio_1y_ago")
    if current_ratio is not None and current_ratio_prev is not None:
        if current_ratio > current_ratio_prev:
            score += 1
    elif current_ratio is not None and current_ratio > 1.5:
        score += 1
    
    # 7. No new shares issued (dilution)
    shares = fundamentals.get("shares_outstanding")
    shares_prev = fundamentals.get("shares_outstanding_prev") or fundamentals.get("shares_1y_ago")
    if shares is not None and shares_prev is not None:
        if shares <= shares_prev:
            score += 1
    else:
        # Assume no dilution if no data
        score += 1
    
    # === OPERATING EFFICIENCY (2 signals) ===
    
    # 8. Improving gross margin
    gross_margin = fundamentals.get("gross_margin")
    gross_margin_prev = fundamentals.get("gross_margin_prev") or fundamentals.get("gross_margin_1y_ago")
    if gross_margin is not None and gross_margin_prev is not None:
        if gross_margin > gross_margin_prev:
            score += 1
    
    # 9. Improving asset turnover (Revenue / Total Assets)
    revenue = fundamentals.get("revenue")
    revenue_prev = fundamentals.get("revenue_prev") or fundamentals.get("revenue_1y_ago")
    
    if revenue is not None and total_assets > 0:
        turnover_curr = revenue / total_assets
        if revenue_prev is not None and assets_prev is not None and assets_prev > 0:
            turnover_prev = revenue_prev / assets_prev
            if turnover_curr > turnover_prev:
                score += 1
    
    return score


def altman_z_score(fundamentals: Dict[str, Any]) -> Optional[float]:
    """
    Calculate Altman Z-Score for bankruptcy prediction.
    
    The Z-Score formula (for manufacturing firms):
    Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    
    Where:
    A = Working Capital / Total Assets
    B = Retained Earnings / Total Assets
    C = EBIT / Total Assets
    D = Market Value of Equity / Total Liabilities
    E = Sales / Total Assets
    
    Interpretation:
    - Z > 2.99: Safe zone (low bankruptcy risk)
    - 1.81 < Z < 2.99: Grey zone (moderate risk)
    - Z < 1.81: Distress zone (high bankruptcy risk)
    
    Args:
        fundamentals: Dictionary with required financial metrics
        
    Returns:
        Z-Score or None if insufficient data
    """
    total_assets = fundamentals.get("total_assets")
    if total_assets is None or total_assets <= 0:
        return None
    
    # A: Working Capital / Total Assets
    current_assets = fundamentals.get("current_assets", 0) or 0
    current_liabilities = fundamentals.get("current_liabilities", 0) or 0
    working_capital = current_assets - current_liabilities
    A = working_capital / total_assets
    
    # B: Retained Earnings / Total Assets
    retained_earnings = fundamentals.get("retained_earnings", 0) or 0
    B = retained_earnings / total_assets
    
    # C: EBIT / Total Assets
    ebit = fundamentals.get("operating_income") or fundamentals.get("ebitda")
    if ebit is None:
        # Approximate EBIT from net income + interest + taxes
        net_income = fundamentals.get("net_income", 0) or 0
        interest = fundamentals.get("interest_expense", 0) or 0
        taxes = fundamentals.get("income_tax_expense", 0) or 0
        ebit = net_income + interest + taxes
    C = ebit / total_assets if ebit else 0
    
    # D: Market Value of Equity / Total Liabilities
    market_cap = fundamentals.get("market_cap")
    total_liabilities = fundamentals.get("total_liabilities")
    if market_cap is not None and total_liabilities is not None and total_liabilities > 0:
        D = market_cap / total_liabilities
    else:
        # Use book equity as fallback
        total_equity = fundamentals.get("total_equity", 0) or 0
        D = total_equity / total_liabilities if total_liabilities and total_liabilities > 0 else 0
    
    # E: Sales / Total Assets (Asset Turnover)
    revenue = fundamentals.get("revenue", 0) or 0
    E = revenue / total_assets
    
    # Calculate Z-Score
    z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
    
    return z_score


def altman_z_score_non_manufacturing(fundamentals: Dict[str, Any]) -> Optional[float]:
    """
    Calculate Altman Z-Score for non-manufacturing/service firms.
    
    Modified formula without the asset turnover component:
    Z = 6.56*A + 3.26*B + 6.72*C + 1.05*D
    
    Interpretation:
    - Z > 2.60: Safe zone
    - 1.10 < Z < 2.60: Grey zone
    - Z < 1.10: Distress zone
    """
    total_assets = fundamentals.get("total_assets")
    if total_assets is None or total_assets <= 0:
        return None
    
    current_assets = fundamentals.get("current_assets", 0) or 0
    current_liabilities = fundamentals.get("current_liabilities", 0) or 0
    working_capital = current_assets - current_liabilities
    A = working_capital / total_assets
    
    retained_earnings = fundamentals.get("retained_earnings", 0) or 0
    B = retained_earnings / total_assets
    
    ebit = fundamentals.get("operating_income") or fundamentals.get("ebitda", 0) or 0
    C = ebit / total_assets
    
    total_equity = fundamentals.get("total_equity", 0) or 0
    total_liabilities = fundamentals.get("total_liabilities", 1) or 1
    D = total_equity / total_liabilities
    
    z_score = 6.56 * A + 3.26 * B + 6.72 * C + 1.05 * D
    
    return z_score


def beneish_m_score(fundamentals: Dict[str, Any]) -> Optional[float]:
    """
    Calculate Beneish M-Score for earnings manipulation detection.
    
    The M-Score is a probabilistic model that uses financial ratios
    to identify potential earnings manipulation.
    
    M-Score > -1.78 suggests higher probability of manipulation.
    
    Components:
    - DSRI: Days Sales in Receivables Index
    - GMI: Gross Margin Index
    - AQI: Asset Quality Index
    - SGI: Sales Growth Index
    - DEPI: Depreciation Index
    - SGAI: SG&A Index
    - LVGI: Leverage Index
    - TATA: Total Accruals to Total Assets
    
    Args:
        fundamentals: Dictionary with current and prior year data
        
    Returns:
        M-Score or None if insufficient data
    """
    # This requires prior year data for comparison
    # Simplified implementation using available ratios
    
    total_assets = fundamentals.get("total_assets")
    if total_assets is None or total_assets <= 0:
        return None
    
    # TATA: Total Accruals to Total Assets
    net_income = fundamentals.get("net_income", 0) or 0
    ocf = fundamentals.get("operating_cashflow", 0) or 0
    tata = (net_income - ocf) / total_assets
    
    # Higher accruals = higher manipulation risk
    # Simplified M-Score approximation based on accruals
    # Full implementation would need YoY comparisons
    
    # Use accruals ratio as proxy
    if abs(tata) > 0.10:  # High accruals
        m_score = -1.5  # Suggests manipulation risk
    elif abs(tata) > 0.05:
        m_score = -2.0  # Grey zone
    else:
        m_score = -2.5  # Lower risk
    
    return m_score


def quality_composite_score(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate composite quality score combining multiple metrics.
    
    Returns:
        Dictionary with individual scores and composite
    """
    f_score = piotroski_f_score(fundamentals)
    z_score = altman_z_score(fundamentals)
    m_score = beneish_m_score(fundamentals)
    
    # Normalize scores to 0-100 scale
    f_normalized = (f_score / 9.0) * 100
    
    z_normalized = None
    if z_score is not None:
        # Z-Score: map < 1.81 to 0, > 2.99 to 100
        z_normalized = max(0, min(100, (z_score - 1.81) / (2.99 - 1.81) * 100))
    
    m_normalized = None
    if m_score is not None:
        # M-Score: < -2.22 is good, > -1.78 is bad
        m_normalized = max(0, min(100, (-m_score - 1.78) / (2.22 - 1.78) * 100))
    
    # Composite (weighted average of available scores)
    scores = [f_normalized]
    weights = [0.5]
    
    if z_normalized is not None:
        scores.append(z_normalized)
        weights.append(0.3)
    
    if m_normalized is not None:
        scores.append(m_normalized)
        weights.append(0.2)
    
    # Normalize weights
    total_weight = sum(weights[:len(scores)])
    weights = [w / total_weight for w in weights[:len(scores)]]
    
    composite = sum(s * w for s, w in zip(scores, weights))
    
    return {
        "f_score": f_score,
        "f_score_normalized": f_normalized,
        "z_score": z_score,
        "z_score_normalized": z_normalized,
        "z_score_zone": _z_score_zone(z_score) if z_score else None,
        "m_score": m_score,
        "m_score_normalized": m_normalized,
        "quality_composite": composite,
        "quality_tier": _quality_tier(composite),
    }


def _z_score_zone(z_score: float) -> str:
    """Classify Z-Score into risk zones."""
    if z_score > 2.99:
        return "safe"
    elif z_score > 1.81:
        return "grey"
    else:
        return "distress"


def _quality_tier(composite: float) -> str:
    """Classify quality composite into tiers."""
    if composite >= 75:
        return "high"
    elif composite >= 50:
        return "medium"
    elif composite >= 25:
        return "low"
    else:
        return "poor"


def compute_quality_features(
    fundamentals_df: pd.DataFrame,
    ticker_col: str = "symbol",
) -> pd.DataFrame:
    """
    Compute quality features for multiple tickers.
    
    Args:
        fundamentals_df: DataFrame with fundamental data
        ticker_col: Column name for ticker
        
    Returns:
        DataFrame with quality features per ticker
    """
    results = []
    
    for ticker, group in fundamentals_df.groupby(ticker_col):
        # Get most recent data
        latest = group.sort_values("date").iloc[-1].to_dict()
        
        # Get prior year data if available
        if len(group) > 4:  # At least 4 quarters
            prior = group.sort_values("date").iloc[-5].to_dict()
            # Add prior year values
            for key in ["roa", "total_debt", "total_assets", "current_ratio", 
                        "shares_outstanding", "gross_margin", "revenue"]:
                if key in prior:
                    latest[f"{key}_prev"] = prior[key]
        
        # Compute scores
        quality = quality_composite_score(latest)
        quality[ticker_col] = ticker
        quality["date"] = latest.get("date")
        
        results.append(quality)
    
    return pd.DataFrame(results)


def get_quality_score(fundamentals: Dict[str, Any]) -> float:
    """
    Get simple quality score (0-100) from fundamentals.
    
    Convenience function for feature engineering.
    """
    quality = quality_composite_score(fundamentals)
    return quality.get("quality_composite", 50.0)
