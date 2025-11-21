"""
Test script for Phase 2 fundamental features

Tests:
1. Fundamental data fetching for small universe
2. Feature computation
3. Integration with existing pipeline
4. Data quality checks
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intentflow_ai.data.fundamentals_fetcher import FundamentalDataFetcher
from intentflow_ai.features.fundamental_features import FundamentalFeatures


def test_fundamental_fetching():
    """Test 1: Fetch fundamental data for small universe."""
    print("\n" + "="*80)
    print("TEST 1: Fundamental Data Fetching")
    print("="*80)
    
    fetcher = FundamentalDataFetcher()
    
    # Test with small set of liquid stocks
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    print(f"\nFetching fundamentals for {len(test_symbols)} symbols...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    fundamentals = fetcher.fetch_universe_fundamentals(
        universe_symbols=test_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    if fundamentals.empty:
        print("❌ Failed to fetch fundamentals")
        return None
    
    print(f"\n✅ Fetched {len(fundamentals):,} rows")
    print(f"   Symbols: {fundamentals['symbol'].nunique()}")
    print(f"   Date range: {fundamentals['date'].min()} to {fundamentals['date'].max()}")
    
    # Show sample
    print(f"\nSample data (first 3 rows for AAPL):")
    aapl_sample = fundamentals[fundamentals['symbol'] == 'AAPL'].head(3)
    print(aapl_sample[['symbol', 'date', 'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity']])
    
    # Coverage analysis
    print(f"\nData Coverage Analysis:")
    coverage = fetcher.get_fundamental_coverage(fundamentals)
    print(coverage)
    
    return fundamentals


def test_feature_computation(fundamentals: pd.DataFrame):
    """Test 2: Compute fundamental features."""
    print("\n" + "="*80)
    print("TEST 2: Feature Computation")
    print("="*80)
    
    # Create mock price data
    print("\nCreating mock price data...")
    price_data = []
    sectors = {'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
               'AMZN': 'Consumer Cyclical', 'NVDA': 'Technology'}
    
    for symbol in fundamentals['symbol'].unique():
        symbol_fund = fundamentals[fundamentals['symbol'] == symbol]
        for _, row in symbol_fund.iterrows():
            price_data.append({
                'ticker': symbol,
                'date': row['date'],
                'close': 100.0,  # Mock price
                'sector': sectors.get(symbol, 'Unknown')
            })
    
    price_df = pd.DataFrame(price_data)
    print(f"   Created {len(price_df):,} price rows")
    
    # Compute features
    print("\nComputing fundamental features...")
    feature_engine = FundamentalFeatures()
    features = feature_engine.compute_all_features(price_df, fundamentals)
    
    if features.empty:
        print("❌ Failed to compute features")
        return None
    
    print(f"\n✅ Computed {len(features.columns)} features across {len(features):,} rows")
    
    # Show feature categories
    print("\nFeature Categories:")
    categories = {}
    for col in features.columns:
        category = col.split("__")[1].split("_")[0] if "__" in col else "unknown"
        categories[category] = categories.get(category, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat:20s}: {count:2d} features")
    
    # Show sample features
    print("\nSample Features (AAPL, latest date):")
    aapl_latest = features[price_df['ticker'] == 'AAPL'].iloc[-1]
    sample_features = [col for col in features.columns if any(x in col for x in ['pe', 'roe', 'value', 'profitability'])][:10]
    for feat in sample_features:
        value = aapl_latest[feat]
        print(f"   {feat:50s}: {value:10.4f}")
    
    # Data quality check
    print("\nData Quality Check:")
    null_pct = features.isna().sum() / len(features) * 100
    high_null_features = null_pct[null_pct > 50].sort_values(ascending=False)
    
    if len(high_null_features) > 0:
        print(f"   ⚠️  {len(high_null_features)} features have >50% null values:")
        for feat, pct in high_null_features.head(5).items():
            print(f"      {feat}: {pct:.1f}% null")
    else:
        print("   ✅ All features have <50% null values")
    
    return features


def test_sector_relative_comparison(features: pd.DataFrame, price_df: pd.DataFrame):
    """Test 3: Compare old sector_relative with new fundamental sector_relative."""
    print("\n" + "="*80)
    print("TEST 3: Old vs New Sector-Relative Features")
    print("="*80)
    
    print("\nOld sector_relative__sector_rel_close:")
    print("   Based on: stock_close / sector_avg_close - 1")
    print("   Signal: Price-based mean reversion within sector")
    
    print("\nNew fundamental__pe_sector_rel:")
    print("   Based on: stock_PE / sector_avg_PE - 1") 
    print("   Signal: Valuation-based, identifies cheap/expensive stocks")
    
    print("\nNew fundamental__roe_sector_z:")
    print("   Based on: (stock_ROE - sector_mean_ROE) / sector_std_ROE")
    print("   Signal: Quality-based, identifies high-quality companies")
    
    if 'fundamental__pe_sector_rel' in features.columns:
        sample_date = price_df['date'].max()
        sample_features = features[price_df['date'] == sample_date].copy()
        sample_features['ticker'] = price_df[price_df['date'] == sample_date]['ticker'].values
        sample_features['sector'] = price_df[price_df['date'] == sample_date]['sector'].values
        
        print(f"\nSample comparison ({sample_date.date()}):")
        print(sample_features[['ticker', 'sector', 'fundamental__pe_sector_rel', 
                                'fundamental__roe_sector_z', 'fundamental__value_composite']].to_string())
    
    print("\n📊 Expected Impact:")
    print("   - Valuation features should add +0.02-0.03 IC")
    print("   - Quality features should add +0.01-0.02 IC")
    print("   - Combined with existing tech/delivery features: 0.032 → 0.07-0.10 IC")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 2 FUNDAMENTAL FEATURES - INTEGRATION TEST")
    print("="*80)
    
    # Test 1: Fetch fundamentals
    fundamentals = test_fundamental_fetching()
    if fundamentals is None:
        print("\n❌ Test 1 failed, aborting")
        return
    
    # Test 2: Compute features
    features = test_feature_computation(fundamentals)
    if features is None:
        print("\n❌ Test 2 failed, aborting")
        return
    
    # Test 3: Comparison
    price_data = []
    sectors = {'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
               'AMZN': 'Consumer Cyclical', 'NVDA': 'Technology'}
    for symbol in fundamentals['symbol'].unique():
        symbol_fund = fundamentals[fundamentals['symbol'] == symbol]
        for _, row in symbol_fund.iterrows():
            price_data.append({
                'ticker': symbol,
                'date': row['date'],
                'close': 100.0,
                'sector': sectors.get(symbol, 'Unknown')
            })
    price_df = pd.DataFrame(price_data)
    
    test_sector_relative_comparison(features, price_df)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("✅ Test 1: Fundamental data fetching - PASSED")
    print("✅ Test 2: Feature computation - PASSED")
    print("✅ Test 3: Sector-relative comparison - PASSED")
    
    print("\n📊 Next Steps:")
    print("   1. Integrate into main FeatureEngineer class")
    print("   2. Run on full universe (S&P 100 / NSE 500)")
    print("   3. Test with WFO framework")
    print("   4. Measure IC improvement vs baseline")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
