"""
Cost model helpers for the backtest.

Provides realistic Indian market transaction cost modeling including:
- Statutory charges (STT, stamp duty, exchange fees, SEBI charges)
- Brokerage (discount broker rates)
- GST on brokerage and exchange fees
- Market impact / slippage (volume-dependent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndianMarketCosts:
    """
    Realistic Indian equity market transaction cost model.
    
    All rates are expressed as fractions (not basis points).
    E.g., 0.001 = 0.1% = 10 bps
    
    Reference rates as of 2024:
    - STT: 0.1% on sell side for delivery
    - Stamp duty: 0.015% on buy side (varies by state)
    - Exchange fees: ~0.00345%
    - SEBI charges: 0.0001%
    - GST: 18% on brokerage + exchange fees
    """
    
    # Brokerage (discount broker like Zerodha)
    brokerage_pct: float = 0.0003  # 0.03% or Rs 20 flat, whichever is lower
    
    # Securities Transaction Tax (STT)
    # 0.1% on sell side for delivery trades
    stt_buy_pct: float = 0.0  # No STT on buy
    stt_sell_pct: float = 0.001  # 0.1% on sell
    
    # Stamp duty (varies by state, using typical rate)
    stamp_duty_buy_pct: float = 0.00015  # 0.015% on buy
    stamp_duty_sell_pct: float = 0.0  # No stamp on sell
    
    # Exchange transaction charges
    exchange_fees_pct: float = 0.0000345  # NSE: 0.00345%
    
    # SEBI turnover fee
    sebi_charges_pct: float = 0.000001  # 0.0001%
    
    # GST on brokerage and exchange fees
    gst_rate: float = 0.18  # 18%
    
    # Base slippage (market impact)
    base_slippage_pct: float = 0.001  # 0.1% base
    
    # Impact coefficient for volume-dependent slippage
    # Higher values = more impact per unit of participation
    impact_coefficient: float = 0.05
    
    def calculate_buy_cost(self, trade_value: float, adv: Optional[float] = None) -> float:
        """
        Calculate total cost for a buy trade.
        
        Args:
            trade_value: Trade value in INR
            adv: Average daily value traded (for slippage calculation)
            
        Returns:
            Total cost in INR
        """
        # Fixed costs
        brokerage = trade_value * self.brokerage_pct
        stamp_duty = trade_value * self.stamp_duty_buy_pct
        exchange_fees = trade_value * self.exchange_fees_pct
        sebi = trade_value * self.sebi_charges_pct
        
        # GST on brokerage and exchange fees
        gst = (brokerage + exchange_fees) * self.gst_rate
        
        # Slippage (market impact)
        slippage = self._calculate_slippage(trade_value, adv)
        
        total = brokerage + stamp_duty + exchange_fees + sebi + gst + slippage
        return total
    
    def calculate_sell_cost(self, trade_value: float, adv: Optional[float] = None) -> float:
        """
        Calculate total cost for a sell trade.
        
        Args:
            trade_value: Trade value in INR
            adv: Average daily value traded (for slippage calculation)
            
        Returns:
            Total cost in INR
        """
        # Fixed costs
        brokerage = trade_value * self.brokerage_pct
        stt = trade_value * self.stt_sell_pct
        exchange_fees = trade_value * self.exchange_fees_pct
        sebi = trade_value * self.sebi_charges_pct
        
        # GST on brokerage and exchange fees
        gst = (brokerage + exchange_fees) * self.gst_rate
        
        # Slippage (market impact)
        slippage = self._calculate_slippage(trade_value, adv)
        
        total = brokerage + stt + exchange_fees + sebi + gst + slippage
        return total
    
    def calculate_round_trip_cost(self, trade_value: float, adv: Optional[float] = None) -> float:
        """
        Calculate total round-trip cost (buy + sell).
        
        Args:
            trade_value: Trade value in INR
            adv: Average daily value traded
            
        Returns:
            Total round-trip cost in INR
        """
        return self.calculate_buy_cost(trade_value, adv) + self.calculate_sell_cost(trade_value, adv)
    
    def calculate_round_trip_pct(self, trade_value: float, adv: Optional[float] = None) -> float:
        """
        Calculate round-trip cost as a percentage.
        
        Args:
            trade_value: Trade value in INR
            adv: Average daily value traded
            
        Returns:
            Round-trip cost as a fraction (e.g., 0.004 = 0.4%)
        """
        if trade_value <= 0:
            return 0.0
        return self.calculate_round_trip_cost(trade_value, adv) / trade_value
    
    def _calculate_slippage(self, trade_value: float, adv: Optional[float] = None) -> float:
        """
        Calculate market impact / slippage using square-root model.
        
        Impact = base_slippage * sqrt(1 + coefficient * participation_rate)
        
        Args:
            trade_value: Trade value in INR
            adv: Average daily value traded
            
        Returns:
            Slippage cost in INR
        """
        if adv is None or adv <= 0:
            # No ADV info, use base slippage
            return trade_value * self.base_slippage_pct
        
        participation_rate = trade_value / adv
        
        # Square-root market impact model
        # This is a standard model used by institutional traders
        impact_multiplier = np.sqrt(1 + self.impact_coefficient * participation_rate * 100)
        slippage_pct = self.base_slippage_pct * impact_multiplier
        
        # Cap slippage at 2% (for extremely illiquid situations)
        slippage_pct = min(slippage_pct, 0.02)
        
        return trade_value * slippage_pct
    
    def get_cost_breakdown(self, trade_value: float, side: str = "buy") -> Dict[str, float]:
        """
        Get detailed breakdown of costs.
        
        Args:
            trade_value: Trade value in INR
            side: "buy" or "sell"
            
        Returns:
            Dictionary with cost components
        """
        breakdown = {
            "trade_value": trade_value,
            "side": side,
            "brokerage": trade_value * self.brokerage_pct,
            "exchange_fees": trade_value * self.exchange_fees_pct,
            "sebi_charges": trade_value * self.sebi_charges_pct,
        }
        
        if side == "buy":
            breakdown["stamp_duty"] = trade_value * self.stamp_duty_buy_pct
            breakdown["stt"] = 0.0
        else:
            breakdown["stamp_duty"] = 0.0
            breakdown["stt"] = trade_value * self.stt_sell_pct
        
        breakdown["gst"] = (breakdown["brokerage"] + breakdown["exchange_fees"]) * self.gst_rate
        breakdown["slippage"] = trade_value * self.base_slippage_pct
        breakdown["total"] = sum(v for k, v in breakdown.items() if k not in ["trade_value", "side"])
        breakdown["total_pct"] = breakdown["total"] / trade_value if trade_value > 0 else 0
        
        return breakdown
    
    def to_bps(self) -> Dict[str, float]:
        """Convert all rates to basis points for reporting."""
        return {
            "brokerage_bps": self.brokerage_pct * 10000,
            "stt_sell_bps": self.stt_sell_pct * 10000,
            "stamp_duty_buy_bps": self.stamp_duty_buy_pct * 10000,
            "exchange_fees_bps": self.exchange_fees_pct * 10000,
            "sebi_charges_bps": self.sebi_charges_pct * 10000,
            "base_slippage_bps": self.base_slippage_pct * 10000,
            "estimated_round_trip_bps": self.calculate_round_trip_pct(100000) * 10000,
        }


# Pre-configured cost models
COST_MODELS = {
    "realistic": IndianMarketCosts(),
    "conservative": IndianMarketCosts(
        base_slippage_pct=0.002,  # 0.2% slippage
        impact_coefficient=0.1,
    ),
    "aggressive": IndianMarketCosts(
        base_slippage_pct=0.0005,  # 0.05% slippage
        impact_coefficient=0.02,
    ),
    "no_slippage": IndianMarketCosts(
        base_slippage_pct=0.0,
        impact_coefficient=0.0,
    ),
}


def get_cost_model(name: str = "realistic") -> IndianMarketCosts:
    """
    Get a pre-configured cost model.
    
    Args:
        name: Model name ("realistic", "conservative", "aggressive", "no_slippage")
        
    Returns:
        Configured IndianMarketCosts instance
    """
    if name not in COST_MODELS:
        available = list(COST_MODELS.keys())
        raise ValueError(f"Unknown cost model '{name}'. Available: {available}")
    return COST_MODELS[name]


def load_cost_model(name: str, path: Path | None = None) -> Dict[str, object]:
    """Load a named transaction-cost model from YAML and compute total bps."""

    cost_path = Path(path) if path else settings.path("config/costs_india.yaml")
    if not cost_path.exists():
        raise FileNotFoundError(f"Cost config not found at {cost_path}")
    data = yaml.safe_load(cost_path.read_text()) or {}
    models = data.get("models", {})
    default_model = data.get("default_model")
    model = models.get(name) or models.get(default_model)
    if model is None:
        raise KeyError(f"Unknown cost model '{name}'. Available: {list(models.keys())}")

    components = model.get("components", {})
    normalized = {}
    total = float(model.get("per_side_bps", 0.0))
    for key, value in components.items():
        try:
            normalized[key] = float(value)
            total += float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Component '{key}' in cost model '{name}' is not numeric.") from exc
    if total <= 0:
        raise ValueError(f"Cost model '{name}' resulted in non-positive total bps.")

    slippage_bps = float(model.get("slippage_bps", data.get("defaults", {}).get("slippage_bps", 0.0)))

    logger.info("Loaded cost model", extra={"cost_model": name, "total_bps": total, "slippage_bps": slippage_bps})
    return {
        "name": name,
        "components": normalized,
        "total_bps": float(total),
        "slippage_bps": slippage_bps,
        "description": model.get("description", ""),
        "source": str(cost_path),
    }


def estimate_annual_cost(
    turnover_pct: float,
    cost_model: IndianMarketCosts | None = None,
    avg_trade_value: float = 100_000,
) -> float:
    """
    Estimate annual transaction costs given turnover.
    
    Args:
        turnover_pct: Annual portfolio turnover as a fraction (e.g., 3.0 = 300%)
        cost_model: Cost model to use (defaults to realistic)
        avg_trade_value: Average trade size in INR
        
    Returns:
        Annual cost as a percentage of portfolio
    """
    if cost_model is None:
        cost_model = get_cost_model("realistic")
    
    # Each unit of turnover requires buy + sell
    round_trip_cost = cost_model.calculate_round_trip_pct(avg_trade_value)
    
    # Annual cost = turnover * round-trip cost
    annual_cost = turnover_pct * round_trip_cost
    
    return annual_cost
