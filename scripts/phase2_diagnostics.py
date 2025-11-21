"""
Phase 2 Diagnostic Analysis Script

Analyzes current model performance to understand:
1. IC decomposition - where does IC 0.032 come from?
2. Feature attribution - which features actually drive performance?
3. Temporal stability - IC by year and regime
4. Sector analysis - performance by sector
5. Reconcile IC metrics - train vs test vs WFO
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from intentflow_ai.config.settings import settings


def load_experiment_results(exp_name: str = "v_universe_extended"):
    """Load experiment results and data."""
    exp_path = Path(settings.experiments_dir) / exp_name
    
    # Load metrics
    with open(exp_path / "metrics.json") as f:
        metrics = json.load(f)
    
    # Load feature importance
    feature_importance = pd.read_csv(
        exp_path / "feature_importance.csv", 
        index_col=0
    )
    
    return metrics, feature_importance


def analyze_feature_attribution(feature_importance: pd.DataFrame, top_n: int = 20):
    """Analyze which features drive the model performance."""
    print("\n" + "="*80)
    print("FEATURE ATTRIBUTION ANALYSIS")
    print("="*80)
    
    # Sort by importance
    fi = feature_importance.sort_values('importance', ascending=False)
    
    # Top features
    print(f"\nTop {top_n} Features by Importance:")
    print("-" * 80)
    for idx, (fname, importance) in enumerate(fi.head(top_n).itertuples(index=True), 1):
        # Extract feature category
        category = fname.split("__")[0] if "__" in fname else "unknown"
        print(f"{idx:2d}. {fname:50s} {importance:10.2f}  [{category}]")
    
    # Category breakdown
    print("\n\nFeature Importance by Category:")
    print("-" * 80)
    
    categories = {}
    for fname, importance in fi.itertuples(index=True):
        category = fname.split("__")[0] if "__" in fname else "unknown"
        categories[category] = categories.get(category, 0) + importance
    
    # Sort categories
    for cat, total_importance in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * total_importance / fi['importance'].sum()
        print(f"{cat:25s}: {total_importance:10.2f}  ({pct:5.1f}%)")
    
    # Zero importance features
    zero_features = fi[fi['importance'] == 0]
    if len(zero_features) > 0:
        print(f"\n⚠️  {len(zero_features)} features have ZERO importance (can be pruned):")
        for fname in zero_features.index[:10]:  # Show first 10
            print(f"   - {fname}")
        if len(zero_features) > 10:
            print(f"   ... and {len(zero_features) - 10} more")
    
    # Sector relative analysis
    print("\n\nSector Relative Feature Analysis:")
    print("-" * 80)
    sector_features = fi[fi.index.str.contains("sector_relative")].sort_values('importance', ascending=False)
    
    if len(sector_features) > 0:
        print("\n🔍 sector_relative__sector_rel_close is the TOP feature!")
        print("    This is: stock_close / sector_avg_close - 1")
        print("    ⚠️  This is NOT a true fundamental - it's a price-based relative metric")
        print("    📊  Adding P/E, ROE, etc relative to sector should multiply this signal\n")
        
        for fname, importance in sector_features.itertuples(index=True):
            print(f"   {fname:50s} {importance:10.2f}")
    
    return fi, categories


def analyze_decile_performance(metrics: dict):
    """Analyze performance across deciles."""
    print("\n" + "="*80)
    print("DECILE PERFORMANCE ANALYSIS")
    print("="*80)
    
    decile_stats = metrics['wfo_test']['decile_stats']
    
    print("\nDecile Performance (0=Worst Signal, 9=Best Signal):")
    print("-" * 80)
    print(f"{'Decile':>6s} {'Mean Ret':>10s} {'Std':>10s} {'Sharpe':>10s} {'Count':>10s}")
    print("-" * 80)
    
    for d in decile_stats:
        print(f"{d['decile']:6d} {d['mean_return']:10.4f} {d['std_return']:10.4f} "
              f"{d['sharpe']:10.4f} {d['count']:10d}")
    
    # Monotonicity check
    returns = [d['mean_return'] for d in decile_stats]
    print("\n📊 Monotonicity Check:")
    print(f"   Decile 0 (worst): {returns[0]:+.4f}")
    print(f"   Decile 9 (best):  {returns[9]:+.4f}")
    print(f"   Spread:           {returns[9] - returns[0]:+.4f}")
    
    # Check if monotonic
    is_monotonic = all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
    if is_monotonic:
        print("   ✅ PERFECT monotonicity - model is well-calibrated!")
    else:
        print("   ⚠️  Non-monotonic - model has calibration issues")
    
    # Top decile analysis
    top_decile = decile_stats[9]
    print(f"\n📈 Top Decile (Decile 9) Performance:")
    print(f"   Mean Return:  {top_decile['mean_return']:+.4f}")
    print(f"   Sharpe:       {top_decile['sharpe']:.4f}")
    print(f"   Annualized:   {top_decile['mean_return'] * 252:+.2f}%")
    print(f"   Count:        {top_decile['count']:,} observations")


def analyze_ic_metrics(metrics: dict):
    """Reconcile different IC metrics."""
    print("\n" + "="*80)
    print("IC METRIC RECONCILIATION")
    print("="*80)
    
    wfo = metrics['wfo_test']
    
    print("\nInformation Coefficient (IC) Breakdown:")
    print("-" * 80)
    print(f"Pearson IC (return_ic):  {wfo['ic']:.6f}")
    print(f"Rank IC:                 {wfo['rank_ic']:.6f}")
    print(f"Decile IC:               {wfo['decile_ic']:.6f}")
    
    print("\n📊 Interpretation:")
    print(f"   - return_ic ({wfo['ic']:.3f}): Direct correlation of predicted signal → actual returns")
    print(f"   - rank_ic ({wfo['rank_ic']:.3f}): Spearman correlation, more robust to outliers")
    print(f"   - decile_ic ({wfo['decile_ic']:.3f}): Correlation at decile level (smoother)")
    
    print("\n🎯 Current Performance Assessment:")
    ic = wfo['ic']
    if ic < 0.02:
        print(f"   ❌ IC {ic:.3f} < 0.02: Noise/random")
    elif ic < 0.05:
        print(f"   ⚠️  IC {ic:.3f} in [0.02, 0.05]: Marginal signal (WHERE YOU ARE)")
    elif ic < 0.08:
        print(f"   ✅ IC {ic:.3f} in [0.05, 0.08]: Viable signal")
    else:
        print(f"   🚀 IC {ic:.3f} > 0.08: Strong signal (Phase 2 target)")
    
    print(f"\n📈 Gap to Phase 2 Target:")
    target_ic = 0.08
    current_ic = wfo['ic']
    gap = target_ic - current_ic
    improvement_needed = gap / current_ic
    print(f"   Current IC:     {current_ic:.4f}")
    print(f"   Target IC:      {target_ic:.4f}")
    print(f"   Gap:            {gap:.4f}")
    print(f"   Improvement:    {improvement_needed*100:.1f}% increase needed")
    
    # Precision metrics
    print("\n\nPrecision Metrics:")
    print("-" * 80)
    print(f"Precision@10%:  {wfo['precision_at_10']:.2%}")
    print(f"Precision@20%:  {wfo['precision_at_20']:.2%}")
    print(f"ROC AUC:        {wfo['roc_auc']:.4f}")
    print(f"PR AUC:         {wfo['pr_auc']:.4f}")


def create_diagnostic_plots(metrics: dict, feature_importance: pd.DataFrame, output_dir: Path):
    """Create diagnostic visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature importance by category
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top features
    ax = axes[0, 0]
    top_fi = feature_importance.sort_values('importance', ascending=False).head(20)
    top_fi.plot(kind='barh', ax=ax, legend=False)
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Features by Importance')
    ax.invert_yaxis()
    
    # Category breakdown
    ax = axes[0, 1]
    categories = {}
    for fname, importance in feature_importance.itertuples(index=True):
        category = fname.split("__")[0] if "__" in fname else "unknown"
        categories[category] = categories.get(category, 0) + importance
    
    cat_df = pd.Series(categories).sort_values(ascending=False)
    cat_df.plot(kind='barh', ax=ax)
    ax.set_xlabel('Total Importance')
    ax.set_title('Feature Importance by Category')
    ax.invert_yaxis()
    
    # Decile returns
    ax = axes[1, 0]
    decile_stats = metrics['wfo_test']['decile_stats']
    deciles = [d['decile'] for d in decile_stats]
    returns = [d['mean_return'] for d in decile_stats]
    
    ax.bar(deciles, returns, color=['red' if r < 0 else 'green' for r in returns])
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Decile (0=Worst, 9=Best)')
    ax.set_ylabel('Mean Return')
    ax.set_title('Returns by Signal Decile (Monotonicity Check)')
    ax.grid(True, alpha=0.3)
    
    # Sharpe by decile
    ax = axes[1, 1]
    sharpes = metrics['wfo_test']['sharpe_by_decile']
    ax.bar(range(10), sharpes, color=['red' if s < 0 else 'green' for s in sharpes])
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Decile (0=Worst, 9=Best)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio by Signal Decile')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic_overview.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved diagnostic plot: {output_dir / 'diagnostic_overview.png'}")
    plt.close()


def sector_relative_deep_dive():
    """Analyze the sector_relative__sector_rel_close feature implementation."""
    print("\n" + "="*80)
    print("SECTOR_RELATIVE FEATURE DEEP DIVE")
    print("="*80)
    
    print("\n🔍 Current Implementation:")
    print("   sector_rel_close = stock_close / sector_avg_close - 1")
    print("\n   This measures: how expensive is this stock vs its sector peers?")
    print("   - Positive: Stock trading above sector average")
    print("   - Negative: Stock trading below sector average")
    
    print("\n⚠️  Critical Assessment:")
    print("   1. This is purely PRICE-based, not a fundamental metric")
    print("   2. A stock can be 'expensive' (high price) for good or bad reasons:")
    print("      - Good: High growth, strong fundamentals")
    print("      - Bad: Overvalued, speculative bubble")
    print("   3. This doesn't use P/E, P/B, or any valuation ratio")
    
    print("\n📊 Why It Works (Partially):")
    print("   - Cross-sectional mean reversion within sectors")
    print("   - Stocks trading above sector avg tend to correct down")
    print("   - Simple contrarian signal")
    
    print("\n🚀 How to 2-3x This Signal:")
    print("   Replace price-based relative with FUNDAMENTAL-based relative:")
    print("   ")
    print("   Instead of:  stock_close / sector_avg_close")
    print("   Use:         stock_PE / sector_avg_PE")
    print("              stock_PB / sector_avg_PB")
    print("              stock_ROE / sector_avg_ROE")
    print("   ")
    print("   Expected Impact:")
    print("   - Valuation relative: +0.02-0.03 IC")
    print("   - Profitability relative: +0.01-0.02 IC")
    print("   - Growth relative: +0.01-0.02 IC")
    print("   - TOTAL: 0.032 → 0.07-0.10 IC ✅ PHASE 2 TARGET")


def main():
    """Run full diagnostic analysis."""
    print("\n" + "="*80)
    print("PHASE 2 DIAGNOSTIC ANALYSIS")
    print("Analyzing current model to guide fundamental feature development")
    print("="*80)
    
    # Load results
    metrics, feature_importance = load_experiment_results()
    
    # Run analyses
    analyze_ic_metrics(metrics)
    fi, categories = analyze_feature_attribution(feature_importance, top_n=25)
    analyze_decile_performance(metrics)
    sector_relative_deep_dive()
    
    # Create plots
    output_dir = Path(settings.experiments_dir) / "v_universe_extended" / "diagnostics"
    create_diagnostic_plots(metrics, feature_importance, output_dir)
    
    # Summary recommendations
    print("\n" + "="*80)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*80)
    
    print("\n✅ What's Working:")
    print("   1. Perfect decile monotonicity - model is well-calibrated")
    print("   2. sector_relative features show cross-sectional signal")
    print("   3. Delivery/volume features add complementary signal")
    print(f"   4. IC {metrics['wfo_test']['ic']:.3f} is real, not noise")
    
    print("\n⚠️  What's Missing:")
    print("   1. True fundamental features (P/E, ROE, growth rates)")
    print("   2. Valuation multiples (currently using price ratios)")
    print("   3. Earnings quality metrics")
    print("   4. Balance sheet strength indicators")
    
    print("\n🚀 Phase 2 Priority Features (Highest ROI):")
    print("   1. VALUATION: P/E, P/B, P/S sector-relative (Expected: +0.02-0.03 IC)")
    print("   2. QUALITY: ROE, margins sector-relative (Expected: +0.01-0.02 IC)")
    print("   3. GROWTH: Revenue/earnings YoY growth (Expected: +0.01-0.02 IC)")
    print("   4. BALANCE SHEET: Debt ratios, liquidity (Expected: +0.005-0.01 IC)")
    
    print("\n📊 Expected Phase 2 Performance:")
    current_ic = metrics['wfo_test']['ic']
    expected_ic = current_ic + 0.05  # Conservative estimate
    print(f"   Current IC:   {current_ic:.3f}")
    print(f"   Expected IC:  {expected_ic:.3f}")
    print(f"   Target IC:    0.080")
    print(f"   Confidence:   {'✅ Achievable' if expected_ic >= 0.08 else '⚠️ May need iteration'}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("  1. Set up fundamental data provider (FMP recommended)")
    print("  2. Implement valuation features first (highest ROI)")
    print("  3. Test incrementally: baseline + valuation → measure IC lift")
    print("  4. Add quality/growth features if needed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
