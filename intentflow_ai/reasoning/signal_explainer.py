"""
Signal Explainer - Human-Readable Trading Signal Explanations

Generates comprehensive explanations for trading signals including:
- Technical analysis summary
- Fundamental analysis summary
- Sentiment/flow analysis summary
- Key SHAP-based drivers
- Risk factors
- Conviction level

Designed for stockbroker decision support.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ConvictionTier(Enum):
    """Signal conviction levels for position sizing."""
    
    HIGH = "HIGH"      # proba > 0.60, allocate 5%
    MEDIUM = "MEDIUM"  # proba > 0.55, allocate 3%
    LOW = "LOW"        # proba > 0.50, allocate 2%
    AVOID = "AVOID"    # proba <= 0.50, don't buy


@dataclass
class SignalExplanation:
    """Complete explanation for a trading signal."""
    
    ticker: str
    sector: str
    date: str
    probability: float
    rank: int
    conviction: ConvictionTier
    
    # Summaries
    technical_summary: str
    fundamental_summary: str
    sentiment_summary: str
    
    # Key drivers from SHAP
    key_drivers: List[Tuple[str, float, str]]  # (feature, importance, interpretation)
    
    # Risk factors
    risk_factors: List[str]
    
    # One-line recommendation
    key_reason: str
    
    # Suggested allocation
    suggested_allocation_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "date": self.date,
            "probability": self.probability,
            "rank": self.rank,
            "conviction": self.conviction.value,
            "technical_summary": self.technical_summary,
            "fundamental_summary": self.fundamental_summary,
            "sentiment_summary": self.sentiment_summary,
            "key_drivers": self.key_drivers,
            "risk_factors": self.risk_factors,
            "key_reason": self.key_reason,
            "suggested_allocation_pct": self.suggested_allocation_pct,
        }
    
    def to_report_string(self) -> str:
        """Generate formatted report string."""
        lines = [
            f"{'='*60}",
            f"RANK #{self.rank}: {self.ticker}",
            f"Sector: {self.sector} | Score: {self.probability:.2f} | Conviction: {self.conviction.value}",
            f"{'='*60}",
            f"",
            f"TECHNICAL: {self.technical_summary}",
            f"FUNDAMENTAL: {self.fundamental_summary}",
            f"SENTIMENT: {self.sentiment_summary}",
            f"",
            f"KEY REASON: {self.key_reason}",
            f"",
        ]
        
        if self.risk_factors:
            lines.append(f"RISKS: {', '.join(self.risk_factors)}")
        
        lines.append(f"ALLOCATION: {self.suggested_allocation_pct:.1f}%")
        lines.append("")
        
        return "\n".join(lines)


class SignalExplainer:
    """
    Generate human-readable explanations for trading signals.
    
    Usage:
        explainer = SignalExplainer()
        explanation = explainer.explain(
            ticker="RELIANCE",
            features=feature_dict,
            shap_values=shap_dict,
            proba=0.68,
            rank=1
        )
        print(explanation.to_report_string())
    """
    
    # Feature name mappings for human-readable output
    FEATURE_NAMES = {
        # Technical
        "rsi_14": "RSI (14-day)",
        "macd": "MACD",
        "macd_signal": "MACD Signal",
        "ema_20": "EMA (20-day)",
        "ema_50": "EMA (50-day)",
        "bb_position": "Bollinger Band Position",
        
        # Momentum
        "ret_5d": "5-day Return",
        "ret_10d": "10-day Return",
        "ret_20d": "20-day Return",
        "momentum_5d": "5-day Momentum",
        "momentum_10d": "10-day Momentum",
        "sector_rel_ret_5d": "Sector-Relative 5d Return",
        
        # Volatility
        "volatility_20d": "20-day Volatility",
        "downside_vol": "Downside Volatility",
        "vol_regime": "Volatility Regime",
        
        # Fundamental
        "pe_ratio": "P/E Ratio",
        "pb_ratio": "P/B Ratio",
        "ps_ratio": "P/S Ratio",
        "roe": "Return on Equity",
        "roa": "Return on Assets",
        "debt_to_equity": "Debt/Equity",
        "current_ratio": "Current Ratio",
        "gross_margin": "Gross Margin",
        "net_margin": "Net Margin",
        "pe_sector_z": "P/E vs Sector",
        "roe_sector_z": "ROE vs Sector",
        
        # Flow/Sentiment
        "fii_flow_5d": "FII Flow (5d)",
        "dii_flow_5d": "DII Flow (5d)",
        "delivery_ratio": "Delivery Ratio",
        "volume_spike": "Volume Spike",
        "pcr": "Put-Call Ratio",
    }
    
    # RSI interpretation thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    def __init__(self):
        pass
    
    def explain(
        self,
        ticker: str,
        features: Dict[str, float],
        proba: float,
        rank: int,
        sector: str = "Unknown",
        date: str = "",
        shap_values: Optional[Dict[str, float]] = None,
    ) -> SignalExplanation:
        """
        Generate comprehensive explanation for a signal.
        
        Args:
            ticker: Stock ticker
            features: Feature values dictionary
            proba: Model probability
            rank: Signal rank
            sector: Stock sector
            date: Signal date
            shap_values: Optional SHAP values for feature importance
            
        Returns:
            SignalExplanation object
        """
        # Determine conviction
        conviction = self._compute_conviction(proba, features, shap_values)
        
        # Generate summaries
        technical_summary = self._technical_summary(features)
        fundamental_summary = self._fundamental_summary(features)
        sentiment_summary = self._sentiment_summary(features)
        
        # Get key drivers from SHAP or feature values
        key_drivers = self._get_key_drivers(features, shap_values)
        
        # Identify risk factors
        risk_factors = self._identify_risks(features, sector)
        
        # Generate key reason
        key_reason = self._generate_key_reason(
            features, shap_values, technical_summary, fundamental_summary
        )
        
        # Suggested allocation based on conviction
        allocation = self._get_allocation(conviction)
        
        return SignalExplanation(
            ticker=ticker,
            sector=sector,
            date=date,
            probability=proba,
            rank=rank,
            conviction=conviction,
            technical_summary=technical_summary,
            fundamental_summary=fundamental_summary,
            sentiment_summary=sentiment_summary,
            key_drivers=key_drivers,
            risk_factors=risk_factors,
            key_reason=key_reason,
            suggested_allocation_pct=allocation,
        )
    
    def _compute_conviction(
        self,
        proba: float,
        features: Dict[str, float],
        shap_values: Optional[Dict[str, float]] = None,
    ) -> ConvictionTier:
        """Compute conviction tier based on probability and feature agreement."""
        
        if proba <= 0.50:
            return ConvictionTier.AVOID
        
        # Check SHAP agreement (if available)
        shap_agreement = 1.0
        if shap_values:
            positive_shap = sum(1 for v in shap_values.values() if v > 0)
            total_shap = len(shap_values)
            if total_shap > 0:
                shap_agreement = positive_shap / total_shap
        
        # High conviction: high probability + high SHAP agreement
        if proba > 0.60 and shap_agreement > 0.6:
            return ConvictionTier.HIGH
        elif proba > 0.55:
            return ConvictionTier.MEDIUM
        else:
            return ConvictionTier.LOW
    
    def _technical_summary(self, features: Dict[str, float]) -> str:
        """Generate technical analysis summary."""
        parts = []
        
        # RSI
        rsi = features.get("rsi_14") or features.get("rsi")
        if rsi is not None:
            if rsi < self.RSI_OVERSOLD:
                parts.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > self.RSI_OVERBOUGHT:
                parts.append(f"RSI overbought ({rsi:.0f})")
            else:
                parts.append(f"RSI neutral ({rsi:.0f})")
        
        # MACD
        macd = features.get("macd")
        macd_signal = features.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                parts.append("MACD bullish crossover")
            else:
                parts.append("MACD bearish")
        
        # Moving averages
        ema_20 = features.get("ema_20")
        ema_50 = features.get("ema_50")
        close = features.get("close")
        if ema_20 and ema_50 and close:
            if close > ema_20 > ema_50:
                parts.append("above key MAs (bullish)")
            elif close < ema_20 < ema_50:
                parts.append("below key MAs (bearish)")
        
        # Momentum
        ret_5d = features.get("ret_5d") or features.get("momentum_5d")
        if ret_5d is not None:
            if ret_5d > 0.05:
                parts.append(f"strong momentum (+{ret_5d*100:.1f}%)")
            elif ret_5d < -0.05:
                parts.append(f"weak momentum ({ret_5d*100:.1f}%)")
        
        if not parts:
            return "No clear technical signals"
        
        return ", ".join(parts)
    
    def _fundamental_summary(self, features: Dict[str, float]) -> str:
        """Generate fundamental analysis summary."""
        parts = []
        
        # P/E ratio
        pe = features.get("pe_ratio")
        pe_sector_z = features.get("pe_sector_z")
        if pe is not None:
            if pe_sector_z is not None and pe_sector_z < -1:
                parts.append(f"P/E {pe:.1f} (cheap vs sector)")
            elif pe_sector_z is not None and pe_sector_z > 1:
                parts.append(f"P/E {pe:.1f} (expensive vs sector)")
            elif pe < 15:
                parts.append(f"P/E {pe:.1f} (value)")
            elif pe > 30:
                parts.append(f"P/E {pe:.1f} (growth)")
        
        # ROE
        roe = features.get("roe")
        roe_sector_z = features.get("roe_sector_z")
        if roe is not None:
            if roe > 0.15:
                parts.append(f"ROE {roe*100:.1f}% (strong)")
            elif roe < 0.08:
                parts.append(f"ROE {roe*100:.1f}% (weak)")
        
        # Debt
        de = features.get("debt_to_equity")
        if de is not None:
            if de > 1.5:
                parts.append(f"D/E {de:.1f} (high leverage)")
            elif de < 0.3:
                parts.append(f"D/E {de:.1f} (low leverage)")
        
        # Margin trends
        margin_trend = features.get("net_margin_trend") or features.get("margin_expansion_4q")
        if margin_trend is not None:
            if margin_trend > 0:
                parts.append("margins expanding")
            elif margin_trend < 0:
                parts.append("margins contracting")
        
        if not parts:
            return "No clear fundamental signals"
        
        return ", ".join(parts)
    
    def _sentiment_summary(self, features: Dict[str, float]) -> str:
        """Generate sentiment/flow analysis summary."""
        parts = []
        
        # FII/DII flows
        fii_flow = features.get("fii_flow_5d") or features.get("fii_change_5d")
        dii_flow = features.get("dii_flow_5d") or features.get("dii_change_5d")
        
        if fii_flow is not None:
            if fii_flow > 0:
                parts.append("FII buying")
            elif fii_flow < 0:
                parts.append("FII selling")
        
        if dii_flow is not None:
            if dii_flow > 0:
                parts.append("DII buying")
        
        # Put-Call Ratio
        pcr = features.get("pcr") or features.get("nifty_pcr")
        if pcr is not None:
            if pcr > 1.2:
                parts.append(f"PCR {pcr:.2f} (contrarian buy)")
            elif pcr < 0.7:
                parts.append(f"PCR {pcr:.2f} (caution)")
        
        # Volume
        volume_spike = features.get("volume_spike") or features.get("volume_z")
        if volume_spike is not None:
            if volume_spike > 2:
                parts.append("unusual volume")
        
        # Delivery
        delivery = features.get("delivery_ratio") or features.get("delivery_z")
        if delivery is not None:
            if delivery > 1:
                parts.append("high delivery (conviction)")
        
        if not parts:
            return "No clear sentiment signals"
        
        return ", ".join(parts)
    
    def _get_key_drivers(
        self,
        features: Dict[str, float],
        shap_values: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float, str]]:
        """Get top 3 key drivers with interpretations."""
        drivers = []
        
        if shap_values:
            # Sort by absolute SHAP value
            sorted_shap = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            for feature, shap_val in sorted_shap:
                feature_name = self.FEATURE_NAMES.get(feature, feature)
                feature_val = features.get(feature)
                
                if shap_val > 0:
                    direction = "positive"
                else:
                    direction = "negative"
                
                interpretation = f"{feature_name}: {direction} impact"
                if feature_val is not None:
                    interpretation += f" (value: {feature_val:.2f})"
                
                drivers.append((feature, shap_val, interpretation))
        else:
            # Fallback: use feature values directly
            key_features = ["rsi_14", "pe_sector_z", "ret_5d", "roe_sector_z", "volume_z"]
            for feat in key_features:
                if feat in features and features[feat] is not None:
                    val = features[feat]
                    feature_name = self.FEATURE_NAMES.get(feat, feat)
                    drivers.append((feat, val, f"{feature_name}: {val:.2f}"))
                    if len(drivers) >= 3:
                        break
        
        return drivers
    
    def _identify_risks(
        self,
        features: Dict[str, float],
        sector: str,
    ) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        
        # High volatility
        vol = features.get("volatility_20d")
        if vol is not None and vol > 0.03:  # >3% daily vol
            risks.append("High volatility")
        
        # High leverage
        de = features.get("debt_to_equity")
        if de is not None and de > 1.5:
            risks.append("High debt levels")
        
        # Overbought
        rsi = features.get("rsi_14") or features.get("rsi")
        if rsi is not None and rsi > 75:
            risks.append("Technically overbought")
        
        # Low liquidity
        volume_z = features.get("volume_z")
        if volume_z is not None and volume_z < -1:
            risks.append("Below-average liquidity")
        
        # Sector-specific risks
        if sector in ["Energy", "Oil & Gas"]:
            risks.append("Oil price sensitivity")
        elif sector in ["Banks", "Financial Services"]:
            risks.append("Interest rate sensitivity")
        elif sector in ["IT", "Technology"]:
            risks.append("USD/INR sensitivity")
        
        # Negative momentum
        ret_20d = features.get("ret_20d")
        if ret_20d is not None and ret_20d < -0.10:
            risks.append("Negative momentum")
        
        return risks[:4]  # Max 4 risks
    
    def _generate_key_reason(
        self,
        features: Dict[str, float],
        shap_values: Optional[Dict[str, float]],
        technical: str,
        fundamental: str,
    ) -> str:
        """Generate one-line key reason for the signal."""
        reasons = []
        
        # Value opportunity
        pe_z = features.get("pe_sector_z")
        if pe_z is not None and pe_z < -0.5:
            reasons.append("undervalued")
        
        # Quality
        roe = features.get("roe")
        if roe is not None and roe > 0.15:
            reasons.append("high quality")
        
        # Momentum
        ret_5d = features.get("ret_5d")
        if ret_5d is not None and ret_5d > 0.03:
            reasons.append("positive momentum")
        
        # Technical
        rsi = features.get("rsi_14") or features.get("rsi")
        if rsi is not None and rsi < 35:
            reasons.append("oversold bounce")
        
        # Smart money
        fii = features.get("fii_flow_5d") or features.get("fii_change_5d")
        if fii is not None and fii > 0:
            reasons.append("smart money accumulating")
        
        if not reasons:
            reasons.append("model signals opportunity")
        
        return ", ".join(reasons).capitalize()
    
    def _get_allocation(self, conviction: ConvictionTier) -> float:
        """Get suggested allocation percentage."""
        allocations = {
            ConvictionTier.HIGH: 5.0,
            ConvictionTier.MEDIUM: 3.0,
            ConvictionTier.LOW: 2.0,
            ConvictionTier.AVOID: 0.0,
        }
        return allocations.get(conviction, 0.0)
    
    def explain_batch(
        self,
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame,
        shap_df: Optional[pd.DataFrame] = None,
    ) -> List[SignalExplanation]:
        """
        Generate explanations for multiple signals.
        
        Args:
            signals_df: DataFrame with [date, ticker, sector, proba, rank]
            features_df: DataFrame with features for each ticker/date
            shap_df: Optional DataFrame with SHAP values
            
        Returns:
            List of SignalExplanation objects
        """
        explanations = []
        
        for _, row in signals_df.iterrows():
            ticker = row["ticker"]
            date = str(row.get("date", ""))
            
            # Get features for this ticker
            features = {}
            ticker_features = features_df[features_df["ticker"] == ticker]
            if not ticker_features.empty:
                features = ticker_features.iloc[-1].to_dict()
            
            # Get SHAP values if available
            shap_values = None
            if shap_df is not None:
                ticker_shap = shap_df[shap_df["ticker"] == ticker]
                if not ticker_shap.empty:
                    shap_values = ticker_shap.iloc[-1].to_dict()
            
            explanation = self.explain(
                ticker=ticker,
                features=features,
                proba=row.get("proba", 0.5),
                rank=int(row.get("rank", 999)),
                sector=row.get("sector", "Unknown"),
                date=date,
                shap_values=shap_values,
            )
            
            explanations.append(explanation)
        
        return explanations


def generate_ranking_report(
    explanations: List[SignalExplanation],
    report_date: str,
    next_rebalance: str,
    top_n: int = 20,
) -> str:
    """
    Generate formatted ranking report for stockbroker use.
    
    Args:
        explanations: List of signal explanations
        report_date: Report generation date
        next_rebalance: Next rebalancing date
        top_n: Number of top signals to include
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "INTENTFLOW AI - Stock Ranking Report",
        f"Generated: {report_date} | Next Rebalance: {next_rebalance}",
        "=" * 70,
        "",
    ]
    
    # Sort by rank
    sorted_explanations = sorted(explanations, key=lambda x: x.rank)[:top_n]
    
    # Summary stats
    high_conviction = sum(1 for e in sorted_explanations if e.conviction == ConvictionTier.HIGH)
    medium_conviction = sum(1 for e in sorted_explanations if e.conviction == ConvictionTier.MEDIUM)
    
    lines.extend([
        "SUMMARY",
        "-" * 70,
        f"Top {top_n} signals | High conviction: {high_conviction} | Medium: {medium_conviction}",
        "",
        "TOP PICKS",
        "-" * 70,
        "",
    ])
    
    # Individual signals
    for explanation in sorted_explanations:
        lines.append(explanation.to_report_string())
    
    # Sector distribution
    sectors = {}
    for e in sorted_explanations:
        sectors[e.sector] = sectors.get(e.sector, 0) + 1
    
    lines.extend([
        "SECTOR DISTRIBUTION",
        "-" * 70,
    ])
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
        lines.append(f"  {sector}: {count} signals")
    
    lines.extend([
        "",
        "=" * 70,
        "Note: This is model-generated advice. Always apply human judgment.",
        "=" * 70,
    ])
    
    return "\n".join(lines)
