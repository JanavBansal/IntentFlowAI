"""
Risk Management Module

Implements drawdown management and exposure control:
- Maximum drawdown stops
- Dynamic exposure reduction
- Stop-loss rules
- Position sizing based on risk

Critical for protecting capital during adverse market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class RiskState(Enum):
    """Current risk state."""
    
    NORMAL = "NORMAL"          # Full exposure allowed
    REDUCED = "REDUCED"        # Reduce exposure to 50%
    MINIMAL = "MINIMAL"        # Reduce exposure to 25%
    EXIT_ALL = "EXIT_ALL"      # Exit all positions


@dataclass
class DrawdownConfig:
    """Configuration for drawdown management."""
    
    # Drawdown thresholds
    max_drawdown: float = 0.15  # 15% max drawdown
    
    # Tiered response thresholds
    tier1_drawdown: float = 0.05  # 5% DD -> alert
    tier2_drawdown: float = 0.10  # 10% DD -> reduce exposure 50%
    tier3_drawdown: float = 0.15  # 15% DD -> exit all
    
    # Recovery thresholds
    recovery_pct: float = 0.50  # Must recover 50% before full exposure
    
    # Stop loss per position
    position_stop_loss: float = 0.08  # 8% stop loss per stock
    
    # Trailing stop
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.12  # 12% trailing stop
    
    # Time-based rules
    min_hold_days: int = 3  # Don't stop out before this
    
    # Volatility adjustment
    vol_adjust_exposure: bool = True
    high_vol_threshold: float = 0.03  # Daily vol > 3%
    high_vol_reduction: float = 0.50  # Reduce to 50% in high vol


@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    
    ticker: str
    entry_date: datetime
    entry_price: float
    current_price: float
    high_since_entry: float
    
    # Computed
    unrealized_pnl_pct: float
    drawdown_from_high: float
    days_held: int
    
    # Flags
    stop_triggered: bool = False
    trailing_stop_triggered: bool = False


class DrawdownManager:
    """
    Manage portfolio drawdown and exposure.
    
    Usage:
        manager = DrawdownManager()
        
        # Update with portfolio value
        manager.update_equity(current_equity)
        
        # Check what exposure is allowed
        exposure_mult = manager.get_allowed_exposure()
        
        # Check individual positions
        positions_to_exit = manager.check_stop_losses(positions)
    """
    
    def __init__(self, config: Optional[DrawdownConfig] = None):
        self.config = config or DrawdownConfig()
        
        # State tracking
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.current_drawdown: float = 0.0
        self.risk_state: RiskState = RiskState.NORMAL
        
        # History
        self.equity_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[Tuple[datetime, float]] = []
        self.state_history: List[Tuple[datetime, RiskState]] = []
    
    def update_equity(self, equity: float, date: Optional[datetime] = None) -> RiskState:
        """
        Update with current portfolio equity.
        
        Args:
            equity: Current portfolio value
            date: Date of update (defaults to now)
            
        Returns:
            Current risk state
        """
        date = date or datetime.now()
        
        self.current_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        
        # Determine risk state
        old_state = self.risk_state
        self.risk_state = self._determine_risk_state()
        
        # Log state change
        if self.risk_state != old_state:
            logger.warning(
                f"Risk state changed: {old_state.value} -> {self.risk_state.value}",
                extra={
                    "drawdown": f"{self.current_drawdown:.2%}",
                    "equity": equity,
                    "peak": self.peak_equity,
                }
            )
            self.state_history.append((date, self.risk_state))
        
        # Record history
        self.equity_history.append((date, equity))
        self.drawdown_history.append((date, self.current_drawdown))
        
        return self.risk_state
    
    def _determine_risk_state(self) -> RiskState:
        """Determine risk state based on drawdown."""
        dd = self.current_drawdown
        cfg = self.config
        
        if dd >= cfg.tier3_drawdown:
            return RiskState.EXIT_ALL
        elif dd >= cfg.tier2_drawdown:
            return RiskState.REDUCED
        elif dd >= cfg.tier1_drawdown:
            return RiskState.MINIMAL
        else:
            return RiskState.NORMAL
    
    def get_allowed_exposure(self, market_vol: Optional[float] = None) -> float:
        """
        Get allowed exposure multiplier based on risk state.
        
        Args:
            market_vol: Optional current market volatility
            
        Returns:
            Exposure multiplier (0.0 to 1.0)
        """
        # Base exposure from risk state
        state_exposure = {
            RiskState.NORMAL: 1.0,
            RiskState.MINIMAL: 0.50,
            RiskState.REDUCED: 0.25,
            RiskState.EXIT_ALL: 0.0,
        }
        
        exposure = state_exposure.get(self.risk_state, 1.0)
        
        # Adjust for volatility
        if self.config.vol_adjust_exposure and market_vol is not None:
            if market_vol > self.config.high_vol_threshold:
                exposure *= self.config.high_vol_reduction
        
        return exposure
    
    def should_reduce_exposure(self) -> Tuple[bool, float]:
        """
        Check if exposure should be reduced.
        
        Returns:
            Tuple of (should_reduce, target_exposure)
        """
        if self.risk_state in [RiskState.REDUCED, RiskState.MINIMAL, RiskState.EXIT_ALL]:
            return True, self.get_allowed_exposure()
        return False, 1.0
    
    def check_stop_losses(
        self,
        positions: List[Dict[str, Any]],
        current_prices: Dict[str, float],
        current_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Check stop loss conditions for positions.
        
        Args:
            positions: List of position dicts with [ticker, entry_price, entry_date, shares]
            current_prices: Current prices per ticker
            current_date: Current date (for days held calculation)
            
        Returns:
            List of tickers that should be exited
        """
        current_date = current_date or datetime.now()
        tickers_to_exit = []
        
        for pos in positions:
            ticker = pos.get("ticker")
            entry_price = pos.get("entry_price", 0)
            entry_date = pos.get("entry_date", current_date)
            high_price = pos.get("high_since_entry", entry_price)
            
            current_price = current_prices.get(ticker, entry_price)
            
            # Update high
            if current_price > high_price:
                high_price = current_price
                pos["high_since_entry"] = high_price
            
            # Calculate metrics
            if entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                dd_from_high = (high_price - current_price) / high_price if high_price > 0 else 0
            else:
                pnl_pct = 0
                dd_from_high = 0
            
            days_held = (current_date - entry_date).days if isinstance(entry_date, datetime) else 0
            
            # Check stop conditions
            should_exit = False
            exit_reason = None
            
            # Fixed stop loss
            if pnl_pct < -self.config.position_stop_loss:
                if days_held >= self.config.min_hold_days:
                    should_exit = True
                    exit_reason = f"Stop loss triggered ({pnl_pct:.1%})"
            
            # Trailing stop
            if self.config.use_trailing_stop and not should_exit:
                if dd_from_high > self.config.trailing_stop_pct:
                    if days_held >= self.config.min_hold_days:
                        should_exit = True
                        exit_reason = f"Trailing stop triggered ({dd_from_high:.1%} from high)"
            
            if should_exit:
                tickers_to_exit.append(ticker)
                logger.info(
                    f"Stop triggered for {ticker}",
                    extra={
                        "reason": exit_reason,
                        "pnl_pct": f"{pnl_pct:.1%}",
                        "days_held": days_held,
                    }
                )
        
        return tickers_to_exit
    
    def compute_position_size(
        self,
        ticker: str,
        signal_strength: float,
        volatility: float,
        portfolio_value: float,
        max_position_pct: float = 0.10,
    ) -> float:
        """
        Compute position size based on risk parameters.
        
        Uses volatility-adjusted position sizing:
        Size = (Target Risk per Position) / (Stock Volatility)
        
        Args:
            ticker: Stock ticker
            signal_strength: Model probability (0-1)
            volatility: Stock's daily volatility
            portfolio_value: Current portfolio value
            max_position_pct: Maximum position size as % of portfolio
            
        Returns:
            Position size in currency units
        """
        # Target risk per position (e.g., 1% of portfolio)
        target_risk_pct = 0.01
        
        # Adjust for current risk state
        exposure_mult = self.get_allowed_exposure()
        
        # Volatility-adjusted size
        if volatility > 0:
            # Inverse volatility sizing
            vol_adjustment = 0.02 / volatility  # Normalize to 2% daily vol
            vol_adjustment = min(max(vol_adjustment, 0.5), 2.0)  # Cap adjustment
        else:
            vol_adjustment = 1.0
        
        # Signal strength adjustment
        signal_adjustment = 0.5 + signal_strength  # 0.5 to 1.5
        
        # Base position size
        base_size = portfolio_value * target_risk_pct * vol_adjustment * signal_adjustment
        
        # Apply exposure multiplier
        size = base_size * exposure_mult
        
        # Cap at max position
        max_size = portfolio_value * max_position_pct
        size = min(size, max_size)
        
        return size
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate risk report."""
        return {
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": f"{self.current_drawdown:.2%}",
            "risk_state": self.risk_state.value,
            "allowed_exposure": f"{self.get_allowed_exposure():.0%}",
            "drawdown_history_30d": [
                {"date": d.isoformat(), "dd": f"{dd:.2%}"}
                for d, dd in self.drawdown_history[-30:]
            ],
        }
    
    def reset(self, initial_equity: float) -> None:
        """Reset manager with new initial equity."""
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.current_drawdown = 0.0
        self.risk_state = RiskState.NORMAL
        self.equity_history = []
        self.drawdown_history = []
        self.state_history = []


def get_drawdown_manager(max_drawdown: float = 0.15) -> DrawdownManager:
    """
    Get configured drawdown manager.
    
    Args:
        max_drawdown: Maximum allowable drawdown (default 15%)
        
    Returns:
        Configured DrawdownManager
    """
    config = DrawdownConfig(
        max_drawdown=max_drawdown,
        tier3_drawdown=max_drawdown,
        tier2_drawdown=max_drawdown * 0.67,  # 2/3 of max
        tier1_drawdown=max_drawdown * 0.33,  # 1/3 of max
    )
    return DrawdownManager(config)
