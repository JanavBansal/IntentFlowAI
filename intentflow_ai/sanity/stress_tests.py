"""Comprehensive stress-testing framework for production trading systems.

Tests strategy robustness under adverse conditions:
- Extreme drawdown scenarios
- High slippage/transaction costs
- Volatility shocks
- Liquidity crises
- Correlation breakdowns
- Parameter sensitivity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    
    # Cost stress tests
    slippage_scenarios: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 50.0, 100.0])  # bps
    fee_scenarios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])  # bps
    
    # Volatility shock scenarios
    vol_multipliers: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0, 5.0])  # Scale returns
    
    # Drawdown scenarios
    simulate_crashes: bool = True
    crash_magnitudes: List[float] = field(default_factory=lambda: [-0.10, -0.20, -0.30, -0.40])  # -10%, -20%, etc.
    
    # Parameter sensitivity
    test_parameter_sensitivity: bool = True
    top_k_range: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30])
    hold_days_range: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    
    # Monte Carlo simulations
    run_monte_carlo: bool = True
    monte_carlo_runs: int = 1000
    bootstrap_block_size: int = 20  # Block bootstrap for time series
    
    # Acceptance criteria
    min_acceptable_sharpe: float = 0.5
    max_acceptable_drawdown: float = -0.30  # -30%
    min_acceptable_win_rate: float = 0.45  # 45%
    
    # Black swan scenarios
    test_black_swans: bool = True
    black_swan_scenarios: List[Dict] = field(default_factory=lambda: [
        {"name": "Flash_Crash", "drop_pct": -0.20, "recovery_days": 5, "affected_pct": 0.80},
        {"name": "Market_Crash", "drop_pct": -0.35, "recovery_days": 20, "affected_pct": 0.95},
        {"name": "Liquidity_Crisis", "spread_multiplier": 5.0, "duration_days": 30},
        {"name": "Correlation_Breakdown", "correlation_shock": True, "duration_days": 60},
        {"name": "Tail_Event", "tail_prob": 0.01, "tail_magnitude": -0.10},
    ])


@dataclass
class StressTestResult:
    """Results from a single stress test scenario."""
    
    scenario_name: str
    scenario_params: Dict
    sharpe: float
    cagr: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario_name,
            "params": self.scenario_params,
            "sharpe": self.sharpe,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }


class StressTestSuite:
    """Comprehensive stress testing for trading strategies."""
    
    def __init__(self, cfg: StressTestConfig = None):
        self.cfg = cfg or StressTestConfig()
        self.results: List[StressTestResult] = []
    
    def run_all_tests(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        base_config: BacktestConfig,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Run comprehensive stress test suite.
        
        Args:
            signals: Trading signals
            prices: Price panel
            base_config: Base backtest configuration
            output_dir: Optional directory to save results
            
        Returns:
            Dict with all stress test results and summary
        """
        logger.info("Starting comprehensive stress test suite")
        
        self.results = []
        
        # 1. Cost stress tests
        cost_results = self._test_cost_scenarios(signals, prices, base_config)
        self.results.extend(cost_results)
        
        # 2. Volatility shock tests
        vol_results = self._test_volatility_shocks(signals, prices, base_config)
        self.results.extend(vol_results)
        
        # 3. Crash scenarios
        if self.cfg.simulate_crashes:
            crash_results = self._test_crash_scenarios(signals, prices, base_config)
            self.results.extend(crash_results)
        
        # 4. Parameter sensitivity
        if self.cfg.test_parameter_sensitivity:
            param_results = self._test_parameter_sensitivity(signals, prices, base_config)
            self.results.extend(param_results)
        
        # 5. Monte Carlo simulation
        if self.cfg.run_monte_carlo:
            mc_results = self._run_monte_carlo(signals, prices, base_config)
        else:
            mc_results = {}
        
        # 6. Black swan scenarios
        if self.cfg.test_black_swans:
            logger.info("Running black swan stress tests...")
            black_swan_results = self._test_black_swans(signals, prices, base_config)
            self.results.extend(black_swan_results)
        
        # Generate summary
        summary = self._generate_summary()
        
        output = {
            "results": [r.to_dict() for r in self.results],
            "summary": summary,
            "monte_carlo": mc_results,
            "config": self.cfg,
        }
        
        if output_dir:
            self._save_results(output, output_dir)
        
        logger.info(
            "Stress test suite complete",
            extra={
                "total_scenarios": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            }
        )
        
        return output
    
    def _test_cost_scenarios(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        base_config: BacktestConfig,
    ) -> List[StressTestResult]:
        """Test sensitivity to transaction costs and slippage."""
        results = []
        
        for slippage in self.cfg.slippage_scenarios:
            for fee in self.cfg.fee_scenarios:
                cfg = self._modify_config(base_config, slippage_bps=slippage, fee_bps=fee)
                
                try:
                    bt_result = backtest_signals(signals, prices, cfg)
                    summary = bt_result["summary"]
                    
                    result = StressTestResult(
                        scenario_name=f"Cost: Slippage={slippage}bps, Fee={fee}bps",
                        scenario_params={"slippage_bps": slippage, "fee_bps": fee},
                        sharpe=summary.get("Sharpe", 0.0),
                        cagr=summary.get("CAGR", 0.0),
                        max_drawdown=summary.get("maxDD", 0.0),
                        win_rate=summary.get("win_rate", 0.0),
                        total_trades=len(bt_result.get("trades", [])),
                        passed=self._check_acceptance_criteria(summary),
                        failure_reasons=self._get_failure_reasons(summary),
                    )
                    results.append(result)
                except Exception as exc:
                    logger.warning(f"Cost scenario failed: {exc}")
        
        return results
    
    def _test_volatility_shocks(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        base_config: BacktestConfig,
    ) -> List[StressTestResult]:
        """Test strategy under amplified volatility."""
        results = []
        
        for vol_mult in self.cfg.vol_multipliers:
            # Amplify price returns
            shocked_prices = self._apply_volatility_shock(prices, vol_mult)
            
            try:
                bt_result = backtest_signals(signals, shocked_prices, base_config)
                summary = bt_result["summary"]
                
                result = StressTestResult(
                    scenario_name=f"Vol Shock: {vol_mult}x",
                    scenario_params={"vol_multiplier": vol_mult},
                    sharpe=summary.get("Sharpe", 0.0),
                    cagr=summary.get("CAGR", 0.0),
                    max_drawdown=summary.get("maxDD", 0.0),
                    win_rate=summary.get("win_rate", 0.0),
                    total_trades=len(bt_result.get("trades", [])),
                    passed=self._check_acceptance_criteria(summary),
                    failure_reasons=self._get_failure_reasons(summary),
                )
                results.append(result)
            except Exception as exc:
                logger.warning(f"Volatility shock scenario failed: {exc}")
        
        return results
    
    def _test_crash_scenarios(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        base_config: BacktestConfig,
    ) -> List[StressTestResult]:
        """Simulate market crashes at random dates."""
        results = []
        
        for crash_mag in self.cfg.crash_magnitudes:
            # Insert crash at mid-point of data
            crashed_prices = self._simulate_crash(prices, crash_mag, crash_date=None)
            
            try:
                bt_result = backtest_signals(signals, crashed_prices, base_config)
                summary = bt_result["summary"]
                
                result = StressTestResult(
                    scenario_name=f"Crash: {crash_mag:.0%} market drop",
                    scenario_params={"crash_magnitude": crash_mag},
                    sharpe=summary.get("Sharpe", 0.0),
                    cagr=summary.get("CAGR", 0.0),
                    max_drawdown=summary.get("maxDD", 0.0),
                    win_rate=summary.get("win_rate", 0.0),
                    total_trades=len(bt_result.get("trades", [])),
                    passed=self._check_acceptance_criteria(summary),
                    failure_reasons=self._get_failure_reasons(summary),
                )
                results.append(result)
            except Exception as exc:
                logger.warning(f"Crash scenario failed: {exc}")
        
        return results
    
    def _test_parameter_sensitivity(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        base_config: BacktestConfig,
    ) -> List[StressTestResult]:
        """Test sensitivity to key parameters (top_k, hold_days)."""
        results = []
        
        for top_k in self.cfg.top_k_range:
            for hold_days in self.cfg.hold_days_range:
                cfg = self._modify_config(base_config, top_k=top_k, hold_days=hold_days)
                
                try:
                    bt_result = backtest_signals(signals, prices, cfg)
                    summary = bt_result["summary"]
                    
                    result = StressTestResult(
                        scenario_name=f"Params: top_k={top_k}, hold={hold_days}d",
                        scenario_params={"top_k": top_k, "hold_days": hold_days},
                        sharpe=summary.get("Sharpe", 0.0),
                        cagr=summary.get("CAGR", 0.0),
                        max_drawdown=summary.get("maxDD", 0.0),
                        win_rate=summary.get("win_rate", 0.0),
                        total_trades=len(bt_result.get("trades", [])),
                        passed=self._check_acceptance_criteria(summary),
                        failure_reasons=self._get_failure_reasons(summary),
                    )
                    results.append(result)
                except Exception as exc:
                    logger.warning(f"Parameter sensitivity scenario failed: {exc}")
        
        return results
    
    def _run_monte_carlo(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        base_config: BacktestConfig,
    ) -> Dict:
        """Run Monte Carlo simulation with block bootstrap.
        
        Randomly resample trading days to assess strategy stability.
        """
        logger.info(f"Running Monte Carlo simulation ({self.cfg.monte_carlo_runs} runs)")
        
        mc_sharpes = []
        mc_drawdowns = []
        mc_win_rates = []
        
        # Get unique dates
        dates = sorted(prices["date"].unique())
        n_dates = len(dates)
        
        if n_dates < self.cfg.bootstrap_block_size:
            logger.warning("Insufficient data for Monte Carlo simulation")
            return {}
        
        for i in range(self.cfg.monte_carlo_runs):
            # Block bootstrap: sample date blocks
            sampled_dates = self._block_bootstrap_dates(dates, self.cfg.bootstrap_block_size)
            
            # Subset prices and signals
            mc_prices = prices[prices["date"].isin(sampled_dates)].copy()
            mc_signals = signals[signals["date"].isin(sampled_dates)].copy()
            
            if mc_prices.empty or mc_signals.empty:
                continue
            
            try:
                bt_result = backtest_signals(mc_signals, mc_prices, base_config)
                summary = bt_result["summary"]
                
                mc_sharpes.append(summary.get("Sharpe", 0.0))
                mc_drawdowns.append(summary.get("maxDD", 0.0))
                mc_win_rates.append(summary.get("win_rate", 0.0))
            except Exception:
                continue
        
        if not mc_sharpes:
            return {}
        
        return {
            "n_runs": len(mc_sharpes),
            "sharpe_mean": float(np.mean(mc_sharpes)),
            "sharpe_std": float(np.std(mc_sharpes)),
            "sharpe_5th_percentile": float(np.percentile(mc_sharpes, 5)),
            "sharpe_95th_percentile": float(np.percentile(mc_sharpes, 95)),
            "max_drawdown_mean": float(np.mean(mc_drawdowns)),
            "max_drawdown_worst": float(np.min(mc_drawdowns)),
            "win_rate_mean": float(np.mean(mc_win_rates)),
            "win_rate_5th_percentile": float(np.percentile(mc_win_rates, 5)),
        }
    
    def _apply_volatility_shock(self, prices: pd.DataFrame, multiplier: float) -> pd.DataFrame:
        """Amplify price volatility by multiplier."""
        shocked = prices.copy()
        
        # Calculate returns and amplify
        shocked = shocked.sort_values(["ticker", "date"])
        shocked["returns"] = shocked.groupby("ticker")["close"].pct_change()
        shocked["shocked_returns"] = shocked["returns"] * multiplier
        
        # Reconstruct prices
        shocked["shocked_close"] = shocked.groupby("ticker")["close"].transform(
            lambda x: x.iloc[0] * (1 + shocked.loc[x.index, "shocked_returns"]).cumprod()
        )
        shocked["close"] = shocked["shocked_close"].fillna(shocked["close"])
        
        return shocked[["date", "ticker", "close"]]
    
    def _simulate_crash(
        self,
        prices: pd.DataFrame,
        magnitude: float,
        crash_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """Simulate a one-day market crash."""
        crashed = prices.copy()
        
        # Default: crash at mid-point
        if crash_date is None:
            dates = sorted(prices["date"].unique())
            crash_date = dates[len(dates) // 2]
        
        # Apply crash to all tickers on crash_date
        crashed.loc[crashed["date"] == crash_date, "close"] *= (1 + magnitude)
        
        # Forward-fill for subsequent dates
        crashed = crashed.sort_values(["ticker", "date"])
        crashed["close"] = crashed.groupby("ticker")["close"].transform(
            lambda x: x.ffill()
        )
        
        return crashed
    
    def _block_bootstrap_dates(self, dates: List, block_size: int) -> List:
        """Block bootstrap to preserve time series structure."""
        n = len(dates)
        n_blocks = n // block_size
        
        sampled = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            block = dates[start_idx:start_idx + block_size]
            sampled.extend(block)
        
        return sampled
    
    def _test_black_swans(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        base_config: BacktestConfig,
    ) -> List[StressTestResult]:
        """Test strategy under black swan scenarios.
        
        Scenarios include:
        - Flash crashes (sudden drops with quick recovery)
        - Market crashes (sustained large drops)
        - Liquidity crises (extreme spreads)
        - Correlation breakdowns (all assets move together)
        - Tail events (extreme outliers)
        """
        results = []
        
        for scenario in self.cfg.black_swan_scenarios:
            scenario_name = scenario["name"]
            logger.info(f"Testing black swan: {scenario_name}")
            
            try:
                # Apply scenario-specific modifications to prices
                if "drop_pct" in scenario:
                    # Flash crash or market crash
                    stressed_prices = self._simulate_crash_scenario(
                        prices.copy(),
                        drop_pct=scenario["drop_pct"],
                        recovery_days=scenario.get("recovery_days", 0),
                        affected_pct=scenario.get("affected_pct", 1.0),
                    )
                elif "spread_multiplier" in scenario:
                    # Liquidity crisis (simulated by increased slippage)
                    stressed_prices = prices.copy()
                    cfg = self._modify_config(
                        base_config,
                        slippage_bps=base_config.slippage_bps * scenario["spread_multiplier"]
                    )
                    base_config = cfg
                elif "correlation_shock" in scenario:
                    # Correlation breakdown (all stocks move together)
                    stressed_prices = self._simulate_correlation_shock(
                        prices.copy(),
                        duration_days=scenario.get("duration_days", 30),
                    )
                elif "tail_prob" in scenario:
                    # Random tail events
                    stressed_prices = self._simulate_tail_events(
                        prices.copy(),
                        tail_prob=scenario["tail_prob"],
                        tail_magnitude=scenario["tail_magnitude"],
                    )
                else:
                    stressed_prices = prices.copy()
                
                # Run backtest on stressed prices
                bt_result = backtest_signals(signals, stressed_prices, base_config)
                summary = bt_result["summary"]
                
                result = StressTestResult(
                    scenario_name=f"Black_Swan_{scenario_name}",
                    scenario_params=scenario,
                    sharpe=summary.get("Sharpe", 0.0),
                    cagr=summary.get("CAGR", 0.0),
                    max_drawdown=summary.get("maxDD", 0.0),
                    win_rate=summary.get("win_rate", 0.0),
                    total_trades=summary.get("total_trades", 0),
                    passed=self._check_acceptance_criteria(summary),
                    failure_reasons=self._get_failure_reasons(summary),
                )
                
                results.append(result)
                logger.debug(f"{scenario_name}: Sharpe={result.sharpe:.2f}, MaxDD={result.max_drawdown:.1%}")
                
            except Exception as exc:
                logger.warning(f"Black swan test failed for {scenario_name}: {exc}")
                continue
        
        return results
    
    def _simulate_crash_scenario(
        self,
        prices: pd.DataFrame,
        drop_pct: float,
        recovery_days: int,
        affected_pct: float,
    ) -> pd.DataFrame:
        """Simulate sudden market crash with optional recovery.
        
        Args:
            prices: Price panel
            drop_pct: Percentage drop (negative value, e.g., -0.20 for -20%)
            recovery_days: Days to recover (0 = no recovery)
            affected_pct: Fraction of stocks affected (0-1)
        """
        prices = prices.copy()
        dates = sorted(prices["date"].unique())
        
        # Crash happens at midpoint
        crash_idx = len(dates) // 2
        crash_date = dates[crash_idx]
        
        # Select affected tickers
        all_tickers = prices["ticker"].unique()
        n_affected = int(len(all_tickers) * affected_pct)
        affected_tickers = np.random.choice(all_tickers, size=n_affected, replace=False)
        
        # Apply crash
        crash_mask = (prices["date"] == crash_date) & (prices["ticker"].isin(affected_tickers))
        prices.loc[crash_mask, "close"] *= (1 + drop_pct)
        
        # Apply recovery over subsequent days
        if recovery_days > 0:
            recovery_per_day = -drop_pct / recovery_days
            
            for i in range(1, recovery_days + 1):
                if crash_idx + i >= len(dates):
                    break
                
                recovery_date = dates[crash_idx + i]
                recovery_mask = (prices["date"] == recovery_date) & (prices["ticker"].isin(affected_tickers))
                prices.loc[recovery_mask, "close"] *= (1 + recovery_per_day)
        
        # Forward-fill to ensure consistency
        prices = prices.sort_values(["ticker", "date"])
        prices["close"] = prices.groupby("ticker")["close"].transform(lambda x: x.ffill())
        
        return prices
    
    def _simulate_correlation_shock(
        self,
        prices: pd.DataFrame,
        duration_days: int = 30,
    ) -> pd.DataFrame:
        """Simulate period where all stocks move together (correlation -> 1).
        
        Simulated by making all stock returns equal to market return.
        """
        prices = prices.copy()
        dates = sorted(prices["date"].unique())
        
        # Select shock period (middle)
        start_idx = len(dates) // 3
        end_idx = start_idx + duration_days
        shock_dates = dates[start_idx:end_idx]
        
        # Compute market return each day
        for date in shock_dates:
            day_prices = prices[prices["date"] == date]
            
            if day_prices.empty:
                continue
            
            # Compute market return (average)
            prev_date_idx = dates.index(date) - 1
            if prev_date_idx < 0:
                continue
            
            prev_date = dates[prev_date_idx]
            prev_prices = prices[prices["date"] == prev_date]
            
            # Align by ticker
            combined = day_prices[["ticker", "close"]].merge(
                prev_prices[["ticker", "close"]],
                on="ticker",
                suffixes=("", "_prev")
            )
            
            if combined.empty:
                continue
            
            # Market return
            market_ret = (combined["close"] / combined["close_prev"]).mean() - 1.0
            
            # Apply same return to all stocks
            for ticker in day_prices["ticker"].unique():
                prev_close = prev_prices[prev_prices["ticker"] == ticker]["close"].values
                if len(prev_close) > 0:
                    new_close = prev_close[0] * (1 + market_ret)
                    prices.loc[(prices["date"] == date) & (prices["ticker"] == ticker), "close"] = new_close
        
        return prices
    
    def _simulate_tail_events(
        self,
        prices: pd.DataFrame,
        tail_prob: float,
        tail_magnitude: float,
    ) -> pd.DataFrame:
        """Inject random extreme events with specified probability.
        
        Args:
            prices: Price panel
            tail_prob: Probability of tail event per stock-day
            tail_magnitude: Size of tail event (negative for drops)
        """
        prices = prices.copy()
        
        # For each row, randomly inject tail events
        n_events = int(len(prices) * tail_prob)
        
        if n_events > 0:
            event_indices = np.random.choice(prices.index, size=n_events, replace=False)
            prices.loc[event_indices, "close"] *= (1 + tail_magnitude)
        
        # Forward-fill to maintain consistency
        prices = prices.sort_values(["ticker", "date"])
        prices["close"] = prices.groupby("ticker")["close"].transform(lambda x: x.ffill())
        
        return prices
    
    def _modify_config(self, base: BacktestConfig, **kwargs) -> BacktestConfig:
        """Create modified backtest config."""
        cfg_dict = {
            "date_col": base.date_col,
            "ticker_col": base.ticker_col,
            "close_col": base.close_col,
            "proba_col": base.proba_col,
            "label_col": base.label_col,
            "hold_days": base.hold_days,
            "top_k": base.top_k,
            "max_weight": base.max_weight,
            "slippage_bps": base.slippage_bps,
            "fee_bps": base.fee_bps,
            "rebalance": base.rebalance,
            "long_only": base.long_only,
            "risk": base.risk,
            "meta": base.meta,
        }
        cfg_dict.update(kwargs)
        return BacktestConfig(**cfg_dict)
    
    def _check_acceptance_criteria(self, summary: Dict) -> bool:
        """Check if results meet minimum acceptance criteria."""
        sharpe = summary.get("Sharpe", 0.0)
        max_dd = summary.get("maxDD", 0.0)
        win_rate = summary.get("win_rate", 0.0)
        
        return (
            sharpe >= self.cfg.min_acceptable_sharpe
            and max_dd >= self.cfg.max_acceptable_drawdown
            and win_rate >= self.cfg.min_acceptable_win_rate
        )
    
    def _get_failure_reasons(self, summary: Dict) -> List[str]:
        """List reasons why scenario failed acceptance criteria."""
        reasons = []
        
        sharpe = summary.get("Sharpe", 0.0)
        max_dd = summary.get("maxDD", 0.0)
        win_rate = summary.get("win_rate", 0.0)
        
        if sharpe < self.cfg.min_acceptable_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < {self.cfg.min_acceptable_sharpe}")
        
        if max_dd < self.cfg.max_acceptable_drawdown:
            reasons.append(f"MaxDD {max_dd:.1%} < {self.cfg.max_acceptable_drawdown:.1%}")
        
        if win_rate < self.cfg.min_acceptable_win_rate:
            reasons.append(f"Win rate {win_rate:.1%} < {self.cfg.min_acceptable_win_rate:.1%}")
        
        return reasons
    
    def _generate_summary(self) -> Dict:
        """Generate summary of all stress tests."""
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]
        
        summary = {
            "total_scenarios": len(self.results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": len(passed) / len(self.results) if self.results else 0.0,
        }
        
        if self.results:
            sharpes = [r.sharpe for r in self.results]
            drawdowns = [r.max_drawdown for r in self.results]
            
            summary.update({
                "sharpe_min": float(np.min(sharpes)),
                "sharpe_mean": float(np.mean(sharpes)),
                "sharpe_max": float(np.max(sharpes)),
                "max_drawdown_worst": float(np.min(drawdowns)),
                "max_drawdown_best": float(np.max(drawdowns)),
            })
        
        return summary
    
    def _save_results(self, output: Dict, output_dir: Path):
        """Save stress test results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        
        # Save summary JSON
        summary_path = output_dir / "stress_test_summary.json"
        with open(summary_path, "w") as f:
            json.dump(output["summary"], f, indent=2, default=float)
        
        # Save detailed results CSV
        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        results_path = output_dir / "stress_test_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save Monte Carlo results if available
        if output.get("monte_carlo"):
            mc_path = output_dir / "monte_carlo_results.json"
            with open(mc_path, "w") as f:
                json.dump(output["monte_carlo"], f, indent=2, default=float)
        
        logger.info("Stress test results saved", extra={"dir": str(output_dir)})


def generate_stress_test_report(
    stress_test_output: Dict,
    output_path: Optional[Path] = None,
) -> str:
    """Generate human-readable stress test report.
    
    Args:
        stress_test_output: Output from StressTestSuite.run_all_tests()
        output_path: Optional path to save markdown report
        
    Returns:
        Report as markdown string
    """
    lines = ["# Stress Test Report\n"]
    
    summary = stress_test_output.get("summary", {})
    
    lines.append("## Summary\n")
    lines.append(f"- **Total scenarios**: {summary.get('total_scenarios', 0)}")
    lines.append(f"- **Passed**: {summary.get('passed', 0)}")
    lines.append(f"- **Failed**: {summary.get('failed', 0)}")
    lines.append(f"- **Pass rate**: {summary.get('pass_rate', 0.0):.1%}\n")
    
    lines.append("## Performance Under Stress\n")
    lines.append(f"- **Sharpe range**: [{summary.get('sharpe_min', 0):.2f}, {summary.get('sharpe_max', 0):.2f}]")
    lines.append(f"- **Mean Sharpe**: {summary.get('sharpe_mean', 0):.2f}")
    lines.append(f"- **Worst drawdown**: {summary.get('max_drawdown_worst', 0):.1%}\n")
    
    # Monte Carlo results
    mc = stress_test_output.get("monte_carlo", {})
    if mc:
        lines.append("## Monte Carlo Simulation\n")
        lines.append(f"- **Runs**: {mc.get('n_runs', 0)}")
        lines.append(f"- **Mean Sharpe**: {mc.get('sharpe_mean', 0):.2f} ± {mc.get('sharpe_std', 0):.2f}")
        lines.append(f"- **5th percentile Sharpe**: {mc.get('sharpe_5th_percentile', 0):.2f}")
        lines.append(f"- **Worst drawdown**: {mc.get('max_drawdown_worst', 0):.1%}\n")
    
    # Failed scenarios
    results = stress_test_output.get("results", [])
    failed = [r for r in results if not r["passed"]]
    if failed:
        lines.append("## Failed Scenarios\n")
        for r in failed[:10]:  # Top 10 failures
            lines.append(f"### {r['scenario']}")
            lines.append(f"- Sharpe: {r['sharpe']:.2f}")
            lines.append(f"- Max DD: {r['max_drawdown']:.1%}")
            lines.append(f"- Win rate: {r['win_rate']:.1%}")
            if r.get("failure_reasons"):
                lines.append(f"- Reasons: {', '.join(r['failure_reasons'])}\n")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Stress test report saved", extra={"path": str(output_path)})
    
    return report

