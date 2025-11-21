"""
Test Screener.in + Hybrid Provider with NSE stocks

Tests the full fundamental features pipeline with real NSE data.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from intentflow_ai.data.fundamentals_provider import get_fundamental_provider
from intentflow_ai.features.fundamental_features import FundamentalFeatures


def test_nse_fetching():
    """Test 1: Fetch fundamentals for NSE stocks."""
    print("\n" + "="*80)
    print("TEST 1: NSE Fundamental Data Fetching (Screener.in)")
    print("="*80)
    
    provider = get_fundamental_provider()
    
    # Test with popular NSE stocks
    test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print(f"\nFetching fundamentals for {len(test_symbols)} NSE symbols...")
    print(f"Symbols: {', '.join(test_symbols)}")
    
    all_data = []
    for symbol in test_symbols:
        print(f"\nFetching {symbol}...")
        df = provider.fetch_fundamentals(symbol, start_date, end_date)
        
        if not df.empty:
            print(f"  ✅ {len(df)} records, columns: {len(df.columns)}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            all_data.append(df)
        else:
            print(f"  ❌ No data")
    
    if not all_data:
        print("\n❌ Failed to fetch any fundamentals")
        return None
    
    fundamentals = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Total: {len(fundamentals):,} rows across {fundamentals['symbol'].nunique()} symbols")
    print(f"\nAvailable columns:")
    for col in sorted(fundamentals.columns):
        non_null = fundamentals[col].notna().sum()
        pct = 100 * non_null / len(fundamentals)
        print(f"  {col:30s}: {pct:5.1f}% coverage")
    
    return fundamentals


def test_feature_computation_nse(fundamentals: pd.DataFrame):
    """Test 2: Compute features for NSE stocks."""
    print("\n" + "="*80)
    print("TEST 2: NSE Feature Computation")
    print("="*80)
    
    # Create mock price data with actual NSE sectors
    print("\nCreating price data with NSE sectors...")
    price_data = []
    
    # NSE sector mapping
    sectors = {
        'RELIANCE': 'Energy',
        'TCS': 'Information Technology',
        'INFY': 'Information Technology',
        'HDFCBANK': 'Financials',
        'ICICIBANK': 'Financials'
    }
    
    for symbol in fundamentals['symbol'].unique():
        symbol_fund = fundamentals[fundamentals['symbol'] == symbol]
        for _, row in symbol_fund.iterrows():
            price_data.append({
                'ticker': symbol,
                'date': row['date'],
                'close': 100.0,  # Mock price
                'sector': sectors.get(symbol, 'Other')
            })
    
    price_df = pd.DataFrame(price_data)
    print(f"  Created {len(price_df):,} price rows")
    
    # Compute features
    print("\nComputing fundamental features...")
    feature_engine = FundamentalFeatures()
    features = feature_engine.compute_all_features(price_df, fundamentals)
    
    if features.empty:
        print("❌ Failed to compute features")
        return None
    
    print(f"\n✅ Computed {len(features.columns)} features")
    
    # Show valuation features for latest date
    print("\nValuation Features (Latest Date, RELIANCE):")
    latest_date = price_df['date'].max()
    reliance_features = features[price_df['ticker'] == 'RELIANCE'].iloc[-1]
    
    val_features = [col for col in features.columns if 'pe_' in col or 'pb_' in col or 'value' in col][:10]
    for feat in val_features:
        value = reliance_features[feat]
        print(f"  {feat:50s}: {value:10.4f}")
    
    # Show sector comparison
    print("\nSector-Relative Features (IT stocks vs Financials):")
    print("\nIT Stocks (TCS, INFY):")
    it_latest = features[(price_df['ticker'].isin(['TCS', 'INFY'])) & (price_df['date'] == latest_date)]
    if not it_latest.empty and 'fundamental__pe_sector_rel' in features.columns:
        print(it_latest[['fundamental__pe_sector_rel', 'fundamental__roe_sector_z']].to_string())
    
    print("\nFinancials (HDFCBANK, ICICIBANK):")
    fin_latest = features[(price_df['ticker'].isin(['HDFCBANK', 'ICICIBANK'])) & (price_df['date'] == latest_date)]
    if not fin_latest.empty and 'fundamental__pe_sector_rel' in features.columns:
        print(fin_latest[['fundamental__pe_sector_rel', 'fundamental__roe_sector_z']].to_string())
    
    return features


def main():
    """Run NSE fundamental features test."""
    print("\n" + "="*80)
    print("NSE FUNDAMENTAL FEATURES - FULL INTEGRATION TEST")
    print("Using Screener.in for Indian Stock Data")
    print("="*80)
    
    # Test 1: Fetch NSE fundamentals
    fundamentals = test_nse_fetching()
    if fundamentals is None:
        print("\n❌ Test 1 failed, aborting")
        return
    
    # Test 2: Compute features
    features = test_feature_computation_nse(fundamentals)
    if features is None:
        print("\n❌ Test 2 failed, aborting")
        return
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("✅ Test 1: NSE data fetching (Screener.in) - PASSED")
    print("✅ Test 2: Feature computation - PASSED")
    
    print("\n📊 Ready for:")
    print("   1. Full NIFTY 200 universe fetch")
    print("   2. Integration into FeatureEngineer")
    print("   3. WFO validation on NSE data")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
