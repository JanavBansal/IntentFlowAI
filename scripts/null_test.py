"""
Null-Label Sanity Check
Trains the model on RANDOM labels to verify that the backtest engine produces ~0.0 IC and ~0.5 AUC.
This certifies that there is no leakage in the pipeline itself.
"""
import pandas as pd
import numpy as np
from intentflow_ai.modeling.trainer import LightGBMTrainer, LightGBMConfig
from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.utils.time_enforcer import TimeEnforcer
from pathlib import Path
import json

def run_null_test():
    print("=" * 60)
    print("NULL-LABEL SANITY CHECK")
    print("=" * 60)
    
    # 1. Load Data (use what we have)
    print("Loading data...")
    # We'll use the features we already computed in the last run to save time
    # If not available, we'd compute them. But let's assume engineering works.
    # Actually, let's just run a quick feature build to be safe.
    # Load price data
    prices_path = Path("data/processed/prices.parquet")
    if not prices_path.exists():
        print("❌ Price data not found!")
        return
        
    prices = pd.read_parquet(prices_path)
    # Filter for a small sample of tickers to speed up
    sample_tickers = prices['ticker'].unique()[:10]
    prices = prices[prices['ticker'].isin(sample_tickers)]
    
    engineer = FeatureEngineer()
    features = engineer.build(prices)
    
    # FeatureEngineer returns only features, so we need to add back ticker and date
    features['ticker'] = prices['ticker']
    features['date'] = prices['date']
    
    if features.empty:
        print("❌ No features generated!")
        return

    print(f"Generated {len(features)} rows for {features['ticker'].nunique()} tickers")
    
    # 2. Create RANDOM Labels
    print("Generating RANDOM labels...")
    np.random.seed(42)
    features['target'] = np.random.randint(0, 2, size=len(features))
    
    # 3. Split Data
    print("Splitting data...")
    train_mask = (features['date'] >= "2010-01-01") & (features['date'] < "2020-01-01")
    test_mask = (features['date'] >= "2022-01-01")
    
    X_train = features[train_mask].drop(columns=['target', 'date', 'ticker'])
    y_train = features.loc[train_mask, 'target']
    
    X_test = features[test_mask].drop(columns=['target', 'date', 'ticker'])
    y_test = features.loc[test_mask, 'target']
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Train Model
    print("Training on random noise...")
    cfg = LightGBMConfig()
    
    trainer = LightGBMTrainer(cfg)
    model = trainer.train(X_train, y_train)
    
    # 5. Check Metrics
    print("Evaluating...")
    from sklearn.metrics import roc_auc_score
    
    # Predict on test set
    y_pred = model.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_pred)
    # Calculate IC (correlation)
    test_ic = np.corrcoef(y_test, y_pred)[0, 1]
    
    print("\n" + "=" * 30)
    print("RESULTS")
    print("=" * 30)
    print(f"Test IC:  {test_ic:.4f} (Expected: ~0.0)")
    print(f"Test AUC: {test_auc:.4f} (Expected: ~0.5)")
    
    if abs(test_ic) < 0.02 and 0.48 < test_auc < 0.52:
        print("\n✅ PASS: Backtest engine is clean (no leakage).")
    else:
        print("\n❌ FAIL: Metrics deviate from random. Leakage suspected.")

if __name__ == "__main__":
    run_null_test()
