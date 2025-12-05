"""
Daily Data Quality Check System

Monitors data quality for:
1. Price data freshness and completeness
2. Fundamental data availability
3. Missing tickers and sectors
4. Anomalous price movements
5. Data staleness alerts

Designed to run daily before model scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class DataQualityLevel(Enum):
    """Data quality assessment levels."""
    
    GOOD = "GOOD"           # All checks pass
    WARNING = "WARNING"     # Minor issues, proceed with caution
    CRITICAL = "CRITICAL"   # Major issues, do not use for trading
    STALE = "STALE"         # Data is outdated


@dataclass
class DataQualityConfig:
    """Configuration for data quality checks."""
    
    # Freshness thresholds
    max_price_staleness_days: int = 3       # Weekend + 1
    max_fundamental_staleness_days: int = 100  # ~1 quarter + buffer
    
    # Completeness thresholds
    min_ticker_coverage: float = 0.90       # At least 90% of universe
    min_price_days: int = 20                # Minimum trading days
    
    # Anomaly thresholds
    max_daily_return: float = 0.30          # 30% single-day move
    max_gap_return: float = 0.50            # 50% overnight gap
    min_volume_threshold: int = 1000        # Minimum volume
    
    # Missing data thresholds
    max_missing_sectors_pct: float = 0.05   # Max 5% unknown sectors
    max_missing_close_pct: float = 0.01     # Max 1% missing closes


@dataclass
class DataQualityReport:
    """Report from data quality check."""
    
    timestamp: datetime
    overall_level: DataQualityLevel
    price_freshness: Dict[str, Any]
    fundamental_freshness: Dict[str, Any]
    completeness: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    missing_data: Dict[str, Any]
    recommendations: List[str]
    can_proceed: bool


class DataQualityChecker:
    """
    Daily data quality monitoring system.
    
    Usage:
        checker = DataQualityChecker()
        report = checker.run_all_checks(price_df, fundamentals_df, universe)
        
        if not report.can_proceed:
            print("Data quality issues detected!")
            for rec in report.recommendations:
                print(f"  - {rec}")
    """
    
    def __init__(self, config: Optional[DataQualityConfig] = None):
        self.config = config or DataQualityConfig()
    
    def run_all_checks(
        self,
        price_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame] = None,
        universe: Optional[Set[str]] = None,
        as_of_date: Optional[datetime] = None,
    ) -> DataQualityReport:
        """
        Run all data quality checks.
        
        Args:
            price_df: Price data with columns [ticker, date, open, high, low, close, volume]
            fundamentals_df: Fundamental data (optional)
            universe: Set of expected tickers
            as_of_date: Reference date for freshness checks
            
        Returns:
            DataQualityReport with findings
        """
        as_of_date = as_of_date or datetime.now()
        
        # Run individual checks
        price_freshness = self._check_price_freshness(price_df, as_of_date)
        fundamental_freshness = self._check_fundamental_freshness(
            fundamentals_df, as_of_date
        ) if fundamentals_df is not None else {"status": "not_checked"}
        
        completeness = self._check_completeness(price_df, universe)
        anomalies = self._check_anomalies(price_df)
        missing_data = self._check_missing_data(price_df)
        
        # Determine overall level
        overall_level, recommendations = self._assess_overall_quality(
            price_freshness,
            fundamental_freshness,
            completeness,
            anomalies,
            missing_data,
        )
        
        can_proceed = overall_level in [DataQualityLevel.GOOD, DataQualityLevel.WARNING]
        
        report = DataQualityReport(
            timestamp=as_of_date,
            overall_level=overall_level,
            price_freshness=price_freshness,
            fundamental_freshness=fundamental_freshness,
            completeness=completeness,
            anomalies=anomalies,
            missing_data=missing_data,
            recommendations=recommendations,
            can_proceed=can_proceed,
        )
        
        self._log_report(report)
        
        return report
    
    def _check_price_freshness(
        self,
        df: pd.DataFrame,
        as_of_date: datetime,
    ) -> Dict[str, Any]:
        """Check if price data is fresh."""
        if df.empty or "date" not in df.columns:
            return {
                "status": "CRITICAL",
                "latest_date": None,
                "days_stale": None,
                "message": "No price data available",
            }
        
        df["date"] = pd.to_datetime(df["date"])
        latest_date = df["date"].max()
        days_stale = (as_of_date - latest_date.to_pydatetime()).days
        
        if days_stale <= self.config.max_price_staleness_days:
            status = "GOOD"
            message = f"Price data is current (last: {latest_date.date()})"
        elif days_stale <= self.config.max_price_staleness_days * 2:
            status = "WARNING"
            message = f"Price data is {days_stale} days old"
        else:
            status = "STALE"
            message = f"Price data is {days_stale} days old - too stale!"
        
        return {
            "status": status,
            "latest_date": str(latest_date.date()),
            "days_stale": days_stale,
            "message": message,
        }
    
    def _check_fundamental_freshness(
        self,
        df: pd.DataFrame,
        as_of_date: datetime,
    ) -> Dict[str, Any]:
        """Check if fundamental data is fresh."""
        if df.empty:
            return {
                "status": "WARNING",
                "latest_date": None,
                "message": "No fundamental data available",
            }
        
        date_col = "available_date" if "available_date" in df.columns else "date"
        if date_col not in df.columns:
            return {"status": "WARNING", "message": "No date column in fundamentals"}
        
        df[date_col] = pd.to_datetime(df[date_col])
        latest_date = df[date_col].max()
        days_stale = (as_of_date - latest_date.to_pydatetime()).days
        
        if days_stale <= self.config.max_fundamental_staleness_days:
            status = "GOOD"
        else:
            status = "WARNING"
        
        return {
            "status": status,
            "latest_date": str(latest_date.date()),
            "days_stale": days_stale,
            "tickers_with_data": df["symbol"].nunique() if "symbol" in df.columns else 0,
        }
    
    def _check_completeness(
        self,
        df: pd.DataFrame,
        universe: Optional[Set[str]],
    ) -> Dict[str, Any]:
        """Check data completeness."""
        if "ticker" not in df.columns:
            return {"status": "WARNING", "message": "No ticker column"}
        
        tickers_in_data = set(df["ticker"].unique())
        
        if universe:
            missing_tickers = universe - tickers_in_data
            coverage = 1 - (len(missing_tickers) / len(universe))
        else:
            missing_tickers = set()
            coverage = 1.0
        
        if coverage >= self.config.min_ticker_coverage:
            status = "GOOD"
        elif coverage >= 0.80:
            status = "WARNING"
        else:
            status = "CRITICAL"
        
        return {
            "status": status,
            "tickers_in_data": len(tickers_in_data),
            "universe_size": len(universe) if universe else None,
            "coverage_pct": coverage * 100,
            "missing_tickers": list(missing_tickers)[:20],  # First 20
            "missing_count": len(missing_tickers),
        }
    
    def _check_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for anomalous price movements."""
        anomalies = []
        
        if not {"ticker", "date", "close"}.issubset(df.columns):
            return anomalies
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"])
        
        # Calculate returns
        df["daily_return"] = df.groupby("ticker")["close"].pct_change()
        
        # Find extreme returns
        extreme_returns = df[abs(df["daily_return"]) > self.config.max_daily_return]
        
        for _, row in extreme_returns.iterrows():
            anomalies.append({
                "type": "extreme_return",
                "ticker": row["ticker"],
                "date": str(row["date"].date()),
                "return": row["daily_return"],
                "close": row["close"],
            })
        
        # Find zero/missing volumes
        if "volume" in df.columns:
            zero_volume = df[df["volume"] < self.config.min_volume_threshold]
            # Only report if many occurrences
            vol_issues = zero_volume.groupby("ticker").size()
            for ticker, count in vol_issues.items():
                if count > 5:  # More than 5 low volume days
                    anomalies.append({
                        "type": "low_volume",
                        "ticker": ticker,
                        "count": count,
                    })
        
        return anomalies[:50]  # Limit to 50 anomalies
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing data patterns."""
        result = {
            "missing_close_pct": 0,
            "missing_sectors_pct": 0,
            "tickers_with_gaps": [],
        }
        
        if "close" in df.columns:
            result["missing_close_pct"] = df["close"].isna().mean() * 100
        
        if "sector" in df.columns:
            unknown_sectors = df["sector"].isna() | (df["sector"] == "Unknown")
            result["missing_sectors_pct"] = unknown_sectors.mean() * 100
        
        # Check for data gaps per ticker
        if {"ticker", "date"}.issubset(df.columns):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            
            for ticker, group in df.groupby("ticker"):
                group = group.sort_values("date")
                date_diff = group["date"].diff()
                large_gaps = date_diff[date_diff > pd.Timedelta(days=10)]
                if len(large_gaps) > 0:
                    result["tickers_with_gaps"].append({
                        "ticker": ticker,
                        "gap_count": len(large_gaps),
                        "max_gap_days": date_diff.max().days,
                    })
        
        return result
    
    def _assess_overall_quality(
        self,
        price_freshness: Dict,
        fundamental_freshness: Dict,
        completeness: Dict,
        anomalies: List,
        missing_data: Dict,
    ) -> Tuple[DataQualityLevel, List[str]]:
        """Assess overall data quality and generate recommendations."""
        recommendations = []
        issues = []
        
        # Check price freshness
        if price_freshness.get("status") == "STALE":
            issues.append("CRITICAL")
            recommendations.append("Price data is too stale. Refresh price data before trading.")
        elif price_freshness.get("status") == "WARNING":
            issues.append("WARNING")
            recommendations.append("Price data is getting old. Consider refreshing.")
        
        # Check completeness
        if completeness.get("status") == "CRITICAL":
            issues.append("CRITICAL")
            recommendations.append(
                f"Only {completeness.get('coverage_pct', 0):.1f}% ticker coverage. "
                "Many tickers missing from data."
            )
        elif completeness.get("status") == "WARNING":
            issues.append("WARNING")
            recommendations.append("Some tickers missing from universe.")
        
        # Check anomalies
        extreme_anomalies = [a for a in anomalies if a.get("type") == "extreme_return"]
        if len(extreme_anomalies) > 10:
            issues.append("WARNING")
            recommendations.append(
                f"Found {len(extreme_anomalies)} extreme price movements. "
                "Check for corporate actions."
            )
        
        # Check missing sectors
        if missing_data.get("missing_sectors_pct", 0) > self.config.max_missing_sectors_pct * 100:
            issues.append("WARNING")
            recommendations.append("Many tickers have unknown sectors. Update sector map.")
        
        # Determine overall level
        if "CRITICAL" in issues:
            overall = DataQualityLevel.CRITICAL
        elif "WARNING" in issues:
            overall = DataQualityLevel.WARNING
        elif price_freshness.get("status") == "STALE":
            overall = DataQualityLevel.STALE
        else:
            overall = DataQualityLevel.GOOD
            recommendations.append("All data quality checks passed.")
        
        return overall, recommendations
    
    def _log_report(self, report: DataQualityReport) -> None:
        """Log the quality report."""
        logger.info(
            f"Data Quality Check: {report.overall_level.value}",
            extra={
                "can_proceed": report.can_proceed,
                "price_status": report.price_freshness.get("status"),
                "completeness": report.completeness.get("coverage_pct"),
                "anomaly_count": len(report.anomalies),
            }
        )
        
        if not report.can_proceed:
            logger.warning("Data quality issues prevent trading!")
            for rec in report.recommendations:
                logger.warning(f"  - {rec}")


def run_daily_check(
    price_file: Optional[str] = None,
    fundamentals_file: Optional[str] = None,
    universe_file: Optional[str] = None,
) -> DataQualityReport:
    """
    Convenience function to run daily data quality check.
    
    Args:
        price_file: Path to price data (defaults to settings)
        fundamentals_file: Path to fundamentals parquet
        universe_file: Path to universe CSV
        
    Returns:
        DataQualityReport
    """
    # Load price data
    if price_file:
        price_path = Path(price_file)
    else:
        price_path = settings.data_dir / "raw" / "price_confirmation" / "all_prices.csv"
    
    if price_path.exists():
        price_df = pd.read_csv(price_path)
    else:
        price_df = pd.DataFrame()
    
    # Load fundamentals
    fundamentals_df = None
    if fundamentals_file:
        fund_path = Path(fundamentals_file)
        if fund_path.exists():
            fundamentals_df = pd.read_parquet(fund_path)
    else:
        fund_path = settings.data_dir / "cache" / "fundamentals" / "eodhd_full.parquet"
        if fund_path.exists():
            fundamentals_df = pd.read_parquet(fund_path)
    
    # Load universe
    universe = None
    if universe_file:
        univ_path = Path(universe_file)
    else:
        univ_path = settings.data_dir / "static" / "sector_map.csv"
    
    if univ_path.exists():
        univ_df = pd.read_csv(univ_path)
        ticker_col = "ticker_nse" if "ticker_nse" in univ_df.columns else "ticker"
        universe = set(univ_df[ticker_col].dropna())
    
    # Run checks
    checker = DataQualityChecker()
    return checker.run_all_checks(price_df, fundamentals_df, universe)
