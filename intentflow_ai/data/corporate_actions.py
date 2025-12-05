"""
Corporate Actions Handler

Handles stock splits, bonuses, and dividends in price data.
Critical for accurate backtesting and feature engineering.

Key functions:
1. Detect suspicious price jumps (potential unhandled splits)
2. Extract split information from EODHD data
3. Adjust historical prices for splits
4. Validate price data integrity
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorporateAction:
    """Represents a corporate action."""
    
    symbol: str
    action_type: str  # "split", "bonus", "dividend"
    date: datetime
    factor: float  # For splits: 2.0 means 2:1 split (price halves)
    details: Optional[str] = None


@dataclass
class CorporateActionsConfig:
    """Configuration for corporate actions handling."""
    
    # Threshold for detecting suspicious price changes
    # A 1-day change > 40% without news is likely a split
    suspicious_change_threshold: float = 0.40
    
    # EODHD data directory
    eodhd_dir: Path = Path("data/raw/eodhd")
    
    # Whether to auto-adjust prices when loading
    auto_adjust: bool = True


class CorporateActionsHandler:
    """
    Handle corporate actions in price data.
    
    Usage:
        handler = CorporateActionsHandler()
        
        # Check for suspicious price jumps
        issues = handler.detect_suspicious_changes(price_df)
        
        # Get split info from EODHD
        splits = handler.get_splits_from_eodhd()
        
        # Adjust prices
        adjusted_df = handler.adjust_prices(price_df, splits)
    """
    
    def __init__(self, config: Optional[CorporateActionsConfig] = None):
        self.config = config or CorporateActionsConfig()
        self._eodhd_splits: Optional[Dict[str, List[CorporateAction]]] = None
    
    def detect_suspicious_changes(
        self, 
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Detect suspicious price changes that might indicate unhandled corporate actions.
        
        Args:
            df: Price DataFrame with columns [date, ticker, close]
            price_col: Column name for price
            
        Returns:
            DataFrame of suspicious changes with columns:
            [date, ticker, prev_close, close, change_pct, likely_action]
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"])
        
        # Calculate daily returns
        df["prev_close"] = df.groupby("ticker")[price_col].shift(1)
        df["change_pct"] = (df[price_col] - df["prev_close"]) / df["prev_close"]
        
        # Find suspicious changes
        threshold = self.config.suspicious_change_threshold
        suspicious = df[
            (df["change_pct"].abs() > threshold) & 
            (df["prev_close"].notna())
        ].copy()
        
        # Classify likely action
        def classify_action(row):
            change = row["change_pct"]
            if change < -0.40:
                # Price dropped significantly - likely split or bonus
                # Check if it's close to common split ratios
                ratios = [0.5, 0.33, 0.25, 0.2, 0.1]  # 2:1, 3:1, 4:1, 5:1, 10:1
                for ratio in ratios:
                    if abs(row[price_col] / row["prev_close"] - ratio) < 0.05:
                        factor = int(1 / ratio)
                        return f"Likely {factor}:1 split"
                return "Possible split/bonus"
            elif change > 0.40:
                # Price jumped significantly - reverse split or error
                return "Possible reverse split or data error"
            return "Large move - verify"
        
        suspicious["likely_action"] = suspicious.apply(classify_action, axis=1)
        
        result = suspicious[["date", "ticker", "prev_close", price_col, "change_pct", "likely_action"]]
        result = result.sort_values("date", ascending=False)
        
        if len(result) > 0:
            logger.warning(
                f"Found {len(result)} suspicious price changes",
                extra={"sample": result.head(5).to_dict("records")}
            )
        
        return result
    
    def get_splits_from_eodhd(self) -> Dict[str, List[CorporateAction]]:
        """
        Extract split information from EODHD JSON files.
        
        Returns:
            Dictionary mapping symbol -> list of CorporateAction
        """
        if self._eodhd_splits is not None:
            return self._eodhd_splits
        
        eodhd_dir = Path(self.config.eodhd_dir)
        if not eodhd_dir.exists():
            logger.warning(f"EODHD directory not found: {eodhd_dir}")
            return {}
        
        splits = {}
        
        for json_path in eodhd_dir.glob("*.json"):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                symbol = json_path.stem
                split_info = data.get("SplitsDividends", {})
                
                # Extract last split (EODHD only provides the most recent)
                last_split_factor = split_info.get("LastSplitFactor")
                last_split_date = split_info.get("LastSplitDate")
                
                if last_split_factor and last_split_date:
                    # Parse factor like "2:1" or "1:10"
                    factor = self._parse_split_factor(last_split_factor)
                    if factor:
                        action = CorporateAction(
                            symbol=symbol,
                            action_type="split",
                            date=pd.to_datetime(last_split_date),
                            factor=factor,
                            details=last_split_factor,
                        )
                        splits[symbol] = [action]
                
            except Exception as e:
                logger.debug(f"Error parsing {json_path.name}: {e}")
        
        self._eodhd_splits = splits
        
        logger.info(f"Extracted splits for {len(splits)} symbols from EODHD")
        return splits
    
    @staticmethod
    def _parse_split_factor(factor_str: str) -> Optional[float]:
        """
        Parse split factor string like "2:1" into a numeric factor.
        
        Returns the factor by which price should be multiplied.
        E.g., "2:1" means 2 new shares for 1 old, so price is halved -> factor = 0.5
        """
        try:
            parts = factor_str.split(":")
            if len(parts) == 2:
                new_shares = float(parts[0])
                old_shares = float(parts[1])
                # Factor is how much to multiply old price to get new price
                return old_shares / new_shares
            return None
        except (ValueError, ZeroDivisionError):
            return None
    
    def adjust_prices(
        self,
        df: pd.DataFrame,
        splits: Optional[Dict[str, List[CorporateAction]]] = None,
        price_cols: List[str] = ["open", "high", "low", "close"],
    ) -> pd.DataFrame:
        """
        Adjust historical prices for splits.
        
        Args:
            df: Price DataFrame with columns [date, ticker, open, high, low, close, volume]
            splits: Dictionary of splits per symbol (uses EODHD if None)
            price_cols: Columns to adjust
            
        Returns:
            DataFrame with adjusted prices
        """
        if splits is None:
            splits = self.get_splits_from_eodhd()
        
        if not splits:
            return df
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        adjusted_count = 0
        
        for symbol, actions in splits.items():
            symbol_mask = df["ticker"] == symbol
            
            for action in actions:
                if action.action_type != "split":
                    continue
                
                # Adjust prices before the split date
                date_mask = df["date"] < action.date
                mask = symbol_mask & date_mask
                
                if mask.any():
                    for col in price_cols:
                        if col in df.columns:
                            df.loc[mask, col] = df.loc[mask, col] * action.factor
                    
                    # Adjust volume inversely
                    if "volume" in df.columns and action.factor > 0:
                        df.loc[mask, "volume"] = df.loc[mask, "volume"] / action.factor
                    
                    adjusted_count += mask.sum()
        
        if adjusted_count > 0:
            logger.info(f"Adjusted {adjusted_count} price records for splits")
        
        return df
    
    def validate_price_data(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> Dict[str, any]:
        """
        Validate price data for common issues.
        
        Checks for:
        1. Suspicious large moves (potential splits)
        2. Stale prices (same close for many days)
        3. Zero or negative prices
        4. Missing data gaps
        
        Args:
            df: Price DataFrame
            price_col: Column to validate
            
        Returns:
            Dictionary with validation results
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        results = {
            "total_records": len(df),
            "unique_tickers": df["ticker"].nunique(),
            "date_range": (df["date"].min(), df["date"].max()),
            "issues": [],
        }
        
        # Check for suspicious moves
        suspicious = self.detect_suspicious_changes(df, price_col)
        if len(suspicious) > 0:
            results["suspicious_moves"] = len(suspicious)
            results["issues"].append(f"{len(suspicious)} suspicious price moves detected")
        
        # Check for zero/negative prices
        invalid_prices = df[df[price_col] <= 0]
        if len(invalid_prices) > 0:
            results["invalid_prices"] = len(invalid_prices)
            results["issues"].append(f"{len(invalid_prices)} zero/negative prices")
        
        # Check for stale prices (same close for 5+ consecutive days)
        df_sorted = df.sort_values(["ticker", "date"])
        df_sorted["price_change"] = df_sorted.groupby("ticker")[price_col].diff()
        df_sorted["is_stale"] = df_sorted["price_change"] == 0
        
        # Count consecutive stale days
        stale_tickers = []
        for ticker, group in df_sorted.groupby("ticker"):
            consecutive = 0
            max_consecutive = 0
            for is_stale in group["is_stale"]:
                if is_stale:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            if max_consecutive >= 5:
                stale_tickers.append((ticker, max_consecutive))
        
        if stale_tickers:
            results["stale_prices"] = stale_tickers[:10]  # Top 10
            results["issues"].append(f"{len(stale_tickers)} tickers with stale prices")
        
        # Check for large gaps in dates
        date_gaps = []
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date")
            gaps = group["date"].diff()
            large_gaps = gaps[gaps > pd.Timedelta(days=10)]
            if len(large_gaps) > 0:
                date_gaps.append(ticker)
        
        if date_gaps:
            results["tickers_with_gaps"] = len(date_gaps)
            results["issues"].append(f"{len(date_gaps)} tickers with >10 day gaps")
        
        results["is_valid"] = len(results["issues"]) == 0
        
        return results
    
    def get_dividend_yield(
        self,
        symbol: str,
    ) -> Optional[float]:
        """Get forward dividend yield for a symbol from EODHD."""
        eodhd_dir = Path(self.config.eodhd_dir)
        json_path = eodhd_dir / f"{symbol}.json"
        
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            return data.get("SplitsDividends", {}).get("ForwardAnnualDividendYield")
        except Exception:
            return None


def get_corporate_actions_handler() -> CorporateActionsHandler:
    """Get configured corporate actions handler."""
    return CorporateActionsHandler()


def audit_price_data(df: pd.DataFrame) -> Dict:
    """
    Quick audit of price data quality.
    
    Args:
        df: Price DataFrame with [date, ticker, close]
        
    Returns:
        Audit results dictionary
    """
    handler = get_corporate_actions_handler()
    return handler.validate_price_data(df)
