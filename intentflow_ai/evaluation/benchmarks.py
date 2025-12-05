"""
Benchmark Comparison Module

Compares strategy performance against various benchmarks:
- NIFTY 50 Total Return
- NIFTY 200 Equal Weight
- Simple Momentum (top N by 6-month return)
- Sector Rotation (top sectors by 1-month return)

Essential for understanding if the model adds value over simple strategies.
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
class BenchmarkConfig:
    """Configuration for benchmark strategies."""
    
    # Momentum strategy
    momentum_lookback_days: int = 126  # 6 months
    momentum_top_n: int = 10
    
    # Sector rotation
    sector_lookback_days: int = 21  # 1 month
    sector_top_n: int = 2
    
    # Rebalancing frequency
    rebalance_days: int = 15  # Semi-monthly
    
    # Risk-free rate for Sharpe calculation
    risk_free_rate: float = 0.06  # 6% for India


@dataclass
class BenchmarkResult:
    """Result of a benchmark strategy."""
    
    name: str
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_return": f"{self.total_return:.2%}",
            "cagr": f"{self.cagr:.2%}",
            "volatility": f"{self.volatility:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "win_rate": f"{self.win_rate:.2%}",
            "num_trades": self.num_trades,
        }


class BenchmarkComparison:
    """
    Compare strategy performance against benchmarks.
    
    Usage:
        comparison = BenchmarkComparison(price_df, sector_map)
        
        # Compute all benchmarks
        benchmarks = comparison.compute_all(
            start_date="2020-01-01",
            end_date="2024-12-31"
        )
        
        # Compare with strategy
        report = comparison.compare_with_strategy(
            strategy_returns,
            benchmarks
        )
    """
    
    def __init__(
        self,
        prices_df: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize benchmark comparison.
        
        Args:
            prices_df: DataFrame with [date, ticker, close] columns
            sector_map: Optional mapping of ticker -> sector
            config: Benchmark configuration
        """
        self.prices = prices_df.copy()
        self.prices["date"] = pd.to_datetime(self.prices["date"])
        self.sector_map = sector_map or {}
        self.config = config or BenchmarkConfig()
    
    def compute_all(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compute all benchmark strategies.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary of benchmark names to results
        """
        results = {}
        
        # NIFTY 50 (buy and hold)
        logger.info("Computing NIFTY 50 benchmark...")
        results["nifty50"] = self._compute_nifty50(start_date, end_date)
        
        # Equal Weight
        logger.info("Computing Equal Weight benchmark...")
        results["equal_weight"] = self._compute_equal_weight(start_date, end_date)
        
        # Momentum
        logger.info("Computing Momentum benchmark...")
        results["momentum"] = self._compute_momentum(start_date, end_date)
        
        # Sector Rotation
        if self.sector_map:
            logger.info("Computing Sector Rotation benchmark...")
            results["sector_rotation"] = self._compute_sector_rotation(start_date, end_date)
        
        return results
    
    def _compute_nifty50(
        self,
        start_date: str,
        end_date: str,
    ) -> BenchmarkResult:
        """Compute NIFTY 50 total return benchmark."""
        # Filter for NIFTY 50 constituents (simplified: use all available)
        df = self.prices[
            (self.prices["date"] >= start_date) &
            (self.prices["date"] <= end_date)
        ].copy()
        
        # Use equal weight of all stocks as proxy
        daily_returns = self._compute_daily_portfolio_return(
            df, 
            weight_method="equal"
        )
        
        return self._compute_metrics(daily_returns, "NIFTY 50 (Proxy)")
    
    def _compute_equal_weight(
        self,
        start_date: str,
        end_date: str,
    ) -> BenchmarkResult:
        """Compute equal-weight portfolio benchmark."""
        df = self.prices[
            (self.prices["date"] >= start_date) &
            (self.prices["date"] <= end_date)
        ].copy()
        
        # Equal weight all stocks, rebalanced at each period
        daily_returns = self._compute_daily_portfolio_return(
            df,
            weight_method="equal",
            rebalance_days=self.config.rebalance_days
        )
        
        return self._compute_metrics(daily_returns, "Equal Weight")
    
    def _compute_momentum(
        self,
        start_date: str,
        end_date: str,
    ) -> BenchmarkResult:
        """
        Compute momentum strategy benchmark.
        
        Buys top N stocks by trailing return.
        """
        df = self.prices[
            (self.prices["date"] >= start_date) &
            (self.prices["date"] <= end_date)
        ].copy()
        
        lookback = self.config.momentum_lookback_days
        top_n = self.config.momentum_top_n
        rebalance = self.config.rebalance_days
        
        # Pivot to wide format
        pivot = df.pivot(index="date", columns="ticker", values="close")
        pivot = pivot.sort_index()
        
        # Calculate returns
        returns = pivot.pct_change()
        
        # Rolling momentum signal (trailing return)
        momentum = pivot / pivot.shift(lookback) - 1
        
        # Generate portfolio returns
        portfolio_returns = []
        rebalance_dates = pivot.index[::rebalance]
        
        holdings = None
        
        for i, date in enumerate(pivot.index):
            if date in rebalance_dates:
                # Select top N by momentum
                mom_scores = momentum.loc[date].dropna()
                if len(mom_scores) >= top_n:
                    top_tickers = mom_scores.nlargest(top_n).index.tolist()
                    # Equal weight among top N
                    holdings = {t: 1.0 / top_n for t in top_tickers}
            
            if holdings:
                # Calculate portfolio return
                day_return = sum(
                    returns.loc[date, t] * w 
                    for t, w in holdings.items()
                    if pd.notna(returns.loc[date].get(t))
                )
                portfolio_returns.append(day_return)
            else:
                portfolio_returns.append(0.0)
        
        ret_series = pd.Series(portfolio_returns, index=pivot.index)
        
        return self._compute_metrics(ret_series, "Momentum Top-10")
    
    def _compute_sector_rotation(
        self,
        start_date: str,
        end_date: str,
    ) -> BenchmarkResult:
        """
        Compute sector rotation strategy benchmark.
        
        Rotates into top performing sectors.
        """
        df = self.prices[
            (self.prices["date"] >= start_date) &
            (self.prices["date"] <= end_date)
        ].copy()
        
        # Add sector
        df["sector"] = df["ticker"].map(self.sector_map)
        df = df.dropna(subset=["sector"])
        
        lookback = self.config.sector_lookback_days
        top_n = self.config.sector_top_n
        rebalance = self.config.rebalance_days
        
        # Calculate sector returns
        sector_returns = df.groupby(["date", "sector"])["close"].mean().unstack()
        sector_returns = sector_returns.pct_change()
        
        # Rolling sector performance
        sector_perf = sector_returns.rolling(lookback).mean()
        
        # Generate portfolio returns
        portfolio_returns = []
        holdings = None
        
        for i, date in enumerate(sector_returns.index):
            if i % rebalance == 0:
                # Select top N sectors
                perf = sector_perf.loc[date].dropna()
                if len(perf) >= top_n:
                    top_sectors = perf.nlargest(top_n).index.tolist()
                    holdings = top_sectors
            
            if holdings:
                # Equal weight among top sectors
                day_return = sector_returns.loc[date, holdings].mean()
                portfolio_returns.append(day_return if pd.notna(day_return) else 0.0)
            else:
                portfolio_returns.append(0.0)
        
        ret_series = pd.Series(portfolio_returns, index=sector_returns.index)
        
        return self._compute_metrics(ret_series, "Sector Rotation")
    
    def _compute_daily_portfolio_return(
        self,
        df: pd.DataFrame,
        weight_method: str = "equal",
        rebalance_days: int = 1,
    ) -> pd.Series:
        """Compute daily portfolio returns."""
        # Pivot to wide format
        pivot = df.pivot(index="date", columns="ticker", values="close")
        
        # Daily returns
        returns = pivot.pct_change()
        
        if weight_method == "equal":
            # Equal weight all available stocks
            portfolio_return = returns.mean(axis=1)
        else:
            portfolio_return = returns.mean(axis=1)
        
        return portfolio_return.fillna(0)
    
    def _compute_metrics(
        self,
        returns: pd.Series,
        name: str,
    ) -> BenchmarkResult:
        """Compute performance metrics from returns series."""
        returns = returns.fillna(0)
        
        # Total return
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        
        # CAGR
        n_years = len(returns) / 252
        if n_years > 0 and total_return > -1:
            cagr = (1 + total_return) ** (1 / n_years) - 1
        else:
            cagr = 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        # Calmar ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return BenchmarkResult(
            name=name,
            total_return=total_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            num_trades=len(returns),
            equity_curve=cumulative,
        )
    
    def compare_with_strategy(
        self,
        strategy_returns: pd.Series,
        benchmarks: Dict[str, BenchmarkResult],
    ) -> pd.DataFrame:
        """
        Compare strategy with benchmarks.
        
        Args:
            strategy_returns: Strategy daily returns
            benchmarks: Dictionary of benchmark results
            
        Returns:
            Comparison DataFrame
        """
        # Compute strategy metrics
        strategy_result = self._compute_metrics(strategy_returns, "Strategy")
        
        # Build comparison table
        rows = [strategy_result.to_dict()]
        for name, result in benchmarks.items():
            rows.append(result.to_dict())
        
        df = pd.DataFrame(rows)
        df = df.set_index("name")
        
        return df


def compute_benchmark_returns(
    prices_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Convenience function to compute benchmark returns.
    
    Args:
        prices_df: Price DataFrame
        start_date: Start date
        end_date: End date
        sector_map: Optional sector mapping
        
    Returns:
        Dictionary of benchmark results
    """
    comparison = BenchmarkComparison(prices_df, sector_map)
    return comparison.compute_all(start_date, end_date)


def generate_comparison_report(
    strategy_returns: pd.Series,
    benchmarks: Dict[str, BenchmarkResult],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate benchmark comparison report.
    
    Args:
        strategy_returns: Strategy daily returns
        benchmarks: Benchmark results
        output_path: Optional file path to save report
        
    Returns:
        Report string
    """
    lines = [
        "=" * 70,
        "BENCHMARK COMPARISON REPORT",
        "=" * 70,
        "",
    ]
    
    # Strategy metrics
    comparison = BenchmarkComparison(pd.DataFrame(), {})
    strategy_result = comparison._compute_metrics(strategy_returns, "Strategy")
    
    lines.extend([
        "STRATEGY PERFORMANCE",
        "-" * 70,
        f"  Total Return: {strategy_result.total_return:.2%}",
        f"  CAGR: {strategy_result.cagr:.2%}",
        f"  Volatility: {strategy_result.volatility:.2%}",
        f"  Sharpe Ratio: {strategy_result.sharpe_ratio:.2f}",
        f"  Max Drawdown: {strategy_result.max_drawdown:.2%}",
        "",
    ])
    
    # Benchmarks
    lines.extend([
        "BENCHMARK COMPARISON",
        "-" * 70,
        f"{'Benchmark':<25} {'Return':<12} {'CAGR':<10} {'Sharpe':<10} {'MaxDD':<10}",
        "-" * 70,
    ])
    
    for name, result in benchmarks.items():
        lines.append(
            f"{result.name:<25} "
            f"{result.total_return:>10.2%} "
            f"{result.cagr:>8.2%} "
            f"{result.sharpe_ratio:>8.2f} "
            f"{result.max_drawdown:>8.2%}"
        )
    
    lines.extend([
        "",
        "OUTPERFORMANCE",
        "-" * 70,
    ])
    
    for name, result in benchmarks.items():
        outperf = strategy_result.cagr - result.cagr
        lines.append(f"  vs {result.name}: {outperf:+.2%} CAGR")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Saved comparison report to {output_path}")
    
    return report
