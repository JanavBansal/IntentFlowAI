#!/usr/bin/env python
"""Backtest Simulator with Regime Filter.

Simulates portfolio performance with:
- Virtual capital (₹10L default)
- Regime-based position sizing
- Transaction costs
- Multiple timeframe testing

Usage:
    python scripts/backtest_simulator.py --capital 1000000 --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeFilter:
    """Regime-based position sizing filter.
    
    Uses market correlation to determine position sizing:
    - Low correlation (< 0.12): Full position (100%)
    - Medium correlation (0.12-0.18): Reduced position (60%)
    - High correlation (> 0.18): Minimal position (30%)
    """
    
    lookback_days: int = 20
    low_threshold: float = 0.12
    high_threshold: float = 0.18
    
    def compute_market_correlation(self, returns: pd.DataFrame) -> float:
        """Compute average pairwise correlation among stocks."""
        if len(returns) < self.lookback_days:
            return 0.15  # Default mid-range
        
        recent = returns.tail(self.lookback_days)
        corr_matrix = recent.corr()
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        return correlations.mean()
    
    def get_position_multiplier(self, correlation: float) -> float:
        """Get position size multiplier based on correlation."""
        if correlation < self.low_threshold:
            return 1.0  # Full position
        elif correlation < self.high_threshold:
            return 0.6  # Reduced position
        else:
            return 0.3  # Minimal position
    
    def get_regime_name(self, correlation: float) -> str:
        """Get human-readable regime name."""
        if correlation < self.low_threshold:
            return "LOW_CORR (Full)"
        elif correlation < self.high_threshold:
            return "MED_CORR (Reduced)"
        else:
            return "HIGH_CORR (Minimal)"


@dataclass
class BacktestResult:
    """Container for backtest results."""
    
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    trades: List[Dict]
    daily_values: pd.Series
    regime_exposure: Dict[str, float]


class BacktestSimulator:
    """Portfolio backtest simulator with regime filtering."""
    
    def __init__(
        self,
        capital: float = 1000000,  # ₹10L default
        top_n: int = 5,  # Number of stocks to hold
        holding_days: int = 15,  # Holding period
        transaction_cost: float = 0.001,  # 0.1% per trade
        use_regime_filter: bool = True,
    ):
        self.capital = capital
        self.top_n = top_n
        self.holding_days = holding_days
        self.transaction_cost = transaction_cost
        self.use_regime_filter = use_regime_filter
        self.regime_filter = RegimeFilter()
    
    def run(
        self,
        prices: pd.DataFrame,
        predictions: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Run backtest simulation.
        
        Args:
            prices: DataFrame with date, ticker, close columns
            predictions: DataFrame with date, ticker, probability columns
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResult with performance metrics
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Filter to date range
        prices = prices[(prices['date'] >= start) & (prices['date'] <= end)].copy()
        predictions = predictions[(predictions['date'] >= start) & (predictions['date'] <= end)].copy()
        
        # Create price matrix for returns
        price_matrix = prices.pivot_table(index='date', columns='ticker', values='close')
        returns = price_matrix.pct_change()
        
        # Trading dates (every holding_days days)
        unique_dates = sorted(prices['date'].unique())
        trade_dates = unique_dates[::self.holding_days]
        
        # Initialize tracking
        portfolio_value = self.capital
        cash = self.capital
        positions = {}  # ticker -> shares
        trades = []
        daily_values = []
        regime_days = {"LOW_CORR": 0, "MED_CORR": 0, "HIGH_CORR": 0}
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: ₹{self.capital:,.0f}")
        logger.info(f"Trade dates: {len(trade_dates)}")
        
        for i, trade_date in enumerate(trade_dates):
            # Get regime
            returns_to_date = returns[returns.index <= trade_date]
            correlation = self.regime_filter.compute_market_correlation(returns_to_date)
            regime = self.regime_filter.get_regime_name(correlation)
            
            if self.use_regime_filter:
                position_mult = self.regime_filter.get_position_multiplier(correlation)
            else:
                position_mult = 1.0
            
            # Track regime exposure
            if "LOW" in regime:
                regime_days["LOW_CORR"] += 1
            elif "MED" in regime:
                regime_days["MED_CORR"] += 1
            else:
                regime_days["HIGH_CORR"] += 1
            
            # Get predictions for this date
            date_preds = predictions[predictions['date'] == trade_date]
            if len(date_preds) == 0:
                continue
            
            # Rank by probability, pick top N
            top_picks = date_preds.nlargest(self.top_n, 'probability')
            
            # Close existing positions
            for ticker, shares in list(positions.items()):
                price_data = prices[(prices['date'] == trade_date) & (prices['ticker'] == ticker)]
                if len(price_data) > 0:
                    close_price = price_data['close'].iloc[0]
                    proceeds = shares * close_price * (1 - self.transaction_cost)
                    cash += proceeds
                    
                    # Record trade
                    trades.append({
                        'date': trade_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares,
                        'price': close_price,
                        'value': proceeds,
                        'regime': regime,
                    })
            
            positions = {}
            
            # Open new positions
            capital_per_stock = (cash * position_mult) / self.top_n
            
            for _, row in top_picks.iterrows():
                ticker = row['ticker']
                price_data = prices[(prices['date'] == trade_date) & (prices['ticker'] == ticker)]
                if len(price_data) == 0:
                    continue
                
                price = price_data['close'].iloc[0]
                shares = int(capital_per_stock / price)
                if shares <= 0:
                    continue
                
                cost = shares * price * (1 + self.transaction_cost)
                if cost > cash:
                    shares = int(cash / (price * (1 + self.transaction_cost)))
                    cost = shares * price * (1 + self.transaction_cost)
                
                if shares > 0:
                    positions[ticker] = shares
                    cash -= cost
                    
                    trades.append({
                        'date': trade_date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': cost,
                        'regime': regime,
                        'probability': row['probability'],
                    })
            
            # Record daily value
            portfolio_value = cash
            for ticker, shares in positions.items():
                price_data = prices[(prices['date'] == trade_date) & (prices['ticker'] == ticker)]
                if len(price_data) > 0:
                    portfolio_value += shares * price_data['close'].iloc[0]
            
            daily_values.append({'date': trade_date, 'value': portfolio_value, 'regime': regime})
            
            logger.info(f"  {trade_date.date()}: Regime={regime}, Value=₹{portfolio_value:,.0f}")
        
        # Calculate final metrics
        daily_df = pd.DataFrame(daily_values)
        if len(daily_df) == 0:
            return BacktestResult(
                initial_capital=self.capital,
                final_capital=self.capital,
                total_return=0,
                annualized_return=0,
                max_drawdown=0,
                sharpe_ratio=0,
                win_rate=0,
                num_trades=0,
                trades=[],
                daily_values=pd.Series(),
                regime_exposure={},
            )
        
        daily_df['returns'] = daily_df['value'].pct_change()
        
        total_return = (portfolio_value / self.capital - 1) * 100
        days = (end - start).days
        annualized_return = ((1 + total_return/100) ** (365/max(days, 1)) - 1) * 100
        
        # Max drawdown
        peak = daily_df['value'].expanding().max()
        drawdown = (daily_df['value'] - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Sharpe ratio (assuming 6% risk-free rate)
        excess_returns = daily_df['returns'] - 0.06/252
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Win rate
        winning_trades = sum(1 for t in trades if t['action'] == 'SELL' and t.get('pnl', 0) > 0)
        total_sells = sum(1 for t in trades if t['action'] == 'SELL')
        win_rate = (winning_trades / total_sells * 100) if total_sells > 0 else 0
        
        # Regime exposure
        total_days = sum(regime_days.values())
        regime_exposure = {k: v/total_days*100 for k, v in regime_days.items()} if total_days > 0 else {}
        
        return BacktestResult(
            initial_capital=self.capital,
            final_capital=portfolio_value,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            num_trades=len(trades),
            trades=trades,
            daily_values=daily_df.set_index('date')['value'],
            regime_exposure=regime_exposure,
        )


def generate_predictions_from_model(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Generate predictions using the trained model."""
    from intentflow_ai.features.engineering import FeatureEngineer
    from intentflow_ai.modeling.ensemble import MultiAlgoEnsemble
    import joblib
    
    # Load latest model if exists
    model_path = ROOT / "experiments" / "v_universe_full" / "model.joblib"
    
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
    else:
        logger.warning("No trained model found, using random predictions for demo")
        # Generate random predictions for demo
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = prices[(prices['date'] >= start) & (prices['date'] <= end)]['date'].unique()
        
        predictions = []
        for date in dates:
            tickers = prices[prices['date'] == date]['ticker'].unique()
            for ticker in tickers:
                predictions.append({
                    'date': date,
                    'ticker': ticker,
                    'probability': np.random.random(),
                })
        
        return pd.DataFrame(predictions)
    
    # Use model to generate predictions
    # ... (implement real prediction logic here)
    
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Backtest Simulator")
    parser.add_argument("--capital", type=float, default=1000000, help="Initial capital (₹)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of stocks to hold")
    parser.add_argument("--holding-days", type=int, default=15, help="Holding period in days")
    parser.add_argument("--use-regime-filter", action="store_true", default=True, help="Use regime filter")
    parser.add_argument("--no-regime-filter", action="store_false", dest="use_regime_filter")
    args = parser.parse_args()
    
    # Load prices
    prices = pd.read_parquet(ROOT / "data" / "processed" / "prices.parquet")
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Generate predictions
    predictions = generate_predictions_from_model(prices, args.start, args.end)
    
    if len(predictions) == 0:
        logger.error("No predictions generated")
        return
    
    # Run backtest
    simulator = BacktestSimulator(
        capital=args.capital,
        top_n=args.top_n,
        holding_days=args.holding_days,
        use_regime_filter=args.use_regime_filter,
    )
    
    result = simulator.run(prices, predictions, args.start, args.end)
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {args.start} to {args.end}")
    print(f"Regime Filter: {'ON' if args.use_regime_filter else 'OFF'}")
    print()
    print(f"Initial Capital: ₹{result.initial_capital:,.0f}")
    print(f"Final Capital:   ₹{result.final_capital:,.0f}")
    print(f"Total Return:    {result.total_return:+.2f}%")
    print(f"Annualized:      {result.annualized_return:+.2f}%")
    print(f"Max Drawdown:    {result.max_drawdown:.2f}%")
    print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"Number of Trades: {result.num_trades}")
    print()
    print("Regime Exposure:")
    for regime, pct in result.regime_exposure.items():
        print(f"  {regime}: {pct:.1f}%")


if __name__ == "__main__":
    main()
