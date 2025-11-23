"""
Run a null-label backtest with random predictions to verify backtest logic.
Expected Sharpe should be ~0.0.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.config.settings import Settings
from intentflow_ai.utils.io import load_price_parquet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running Null-Label Test (Seed: {args.seed})...")
    
    # 1. Load Prices
    cfg = Settings()
    prices = load_price_parquet(cfg=cfg)
    print(f"Loaded prices for {len(prices['ticker'].unique())} tickers.")

    # 2. Generate Random Predictions
    # We'll generate predictions for all tickers on all dates in prices
    # But to be realistic, we should probably just use the price index
    
    # Create a DataFrame of all ticker-date combinations present in prices
    # prices usually has 'date', 'ticker', 'close', etc.
    
    # Let's just take unique dates and tickers from prices
    dates = prices['date'].unique()
    tickers = prices['ticker'].unique()
    
    print("Generating random predictions...")
    np.random.seed(args.seed)
    
    # We can just sample from prices to get valid date-ticker pairs
    # This is faster than cross-product
    preds = prices[['date', 'ticker']].copy()
    preds['proba'] = np.random.rand(len(preds))
    preds['label'] = np.random.randint(0, 2, len(preds)) # Dummy label
    
    # 3. Run Backtest
    print("Running backtest...")
    bt_cfg = BacktestConfig(
        top_k=10,
        hold_days=10,
        slippage_bps=10.0, # Standard assumption
        fee_bps=10.0,      # Standard assumption
    )
    
    result = backtest_signals(preds, prices, bt_cfg)
    summary = result['summary']
    
    print("\n--- Null-Label Test Results ---")
    print(f"CAGR: {summary['CAGR']:.2%}")
    print(f"Sharpe: {summary['Sharpe']:.4f}")
    print(f"MaxDD: {summary['maxDD']:.2%}")
    print(f"Win Rate: {summary['win_rate']:.2%}")
    
    if abs(summary['Sharpe']) < 0.5:
        print("\n[PASS] Sharpe is close to 0.0 (within +/- 0.5 tolerance for noise).")
    else:
        print("\n[FAIL] Sharpe is too high/low for random predictions! Check backtest logic.")

if __name__ == "__main__":
    main()
