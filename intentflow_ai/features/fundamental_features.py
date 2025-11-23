"""
Fundamental Features Module

Implements fundamental feature engineering for Phase 2:
- Valuation metrics (P/E, P/B, P/S) with sector-relative transformations
- Profitability metrics (ROE, ROA, margins) with sector-relative
- Growth metrics (revenue, earnings YoY)
- Balance sheet metrics (debt ratios, liquidity)
- Earnings quality metrics (accruals, cash flow quality)

All features maintain point-in-time correctness via reporting delay.
"""

from typing import Optional

import numpy as np
import pandas as pd


class FundamentalFeatures:
    """Fundamental feature engineering with sector-relative transformations."""
    
    def __init__(self, sector_map: Optional[pd.DataFrame] = None):
        """
        Initialize fundamental feature engineer.
        
        Args:
            sector_map: DataFrame mapping symbols to sectors
        """
        self.sector_map = sector_map
    
    def compute_all_features(
        self, 
        price_data: pd.DataFrame,
        fundamental_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute all fundamental features.
        
        Args:
            price_data: Price/volume data with columns [ticker, date, close, ...]
            fundamental_data: Fundamental data from provider
            
        Returns:
            DataFrame with fundamental features
        """
        # Merge fundamentals with price data using point-in-time correctness
        from intentflow_ai.utils.time_enforcer import TimeEnforcer
        
        # Ensure both are sorted by date
        price_data = price_data.sort_values('date')
        
        # Prepare fundamentals for merge
        fund_merge = fundamental_data.copy()
        
        # Ensure available_date exists and is correct
        if 'available_date' not in fund_merge.columns:
             if 'report_date' in fund_merge.columns:
                 fund_merge = TimeEnforcer.apply_reporting_delay(
                     fund_merge, 'report_date', delay_days=45, output_col='available_date'
                 )
             else:
                 # Fallback: assume date is report date
                 fund_merge = TimeEnforcer.apply_reporting_delay(
                     fund_merge, 'date', delay_days=90, output_col='available_date'
                 )
        
        # Rename symbol to ticker for consistency
        if 'symbol' in fund_merge.columns and 'ticker' not in fund_merge.columns:
            fund_merge = fund_merge.rename(columns={'symbol': 'ticker'})
            
        # Drop rows with missing available_date (cannot merge)
        if 'available_date' in fund_merge.columns:
            fund_merge = fund_merge.dropna(subset=['available_date'])
            fund_merge['available_date'] = pd.to_datetime(fund_merge['available_date'])
            fund_merge = fund_merge.sort_values('available_date')
            
        # Perform strict point-in-time merge
        # This ensures we ONLY see fundamental data where available_date <= price_date
        df = TimeEnforcer.merge_asof_safe(
            left=price_data,
            right=fund_merge,
            left_date_col='date',
            right_date_col='available_date',
            by='ticker',
            suffixes=('', '_fund')
        )
        
        df = df.sort_values(['ticker', 'date'])
        
        # FORWARD FILL FUNDAMENTALS
        # Fundamentals are quarterly/annual, prices are daily.
        # merge_asof gives us the latest fundamental row for each price row, but subsequent
        # daily rows for the same stock will have NaNs for fundamental columns if not merged correctly.
        # Actually, merge_asof matches to the *closest previous* row. So every price row gets a match.
        # However, if there is NO previous fundamental row (e.g. start of history), it will be NaN.
        
        # Let's check the fundamental columns.
        # The issue might be that `merge_asof` logic in TimeEnforcer is correct, but
        # the column mapping or something is failing.
        
        # Wait, TimeEnforcer.merge_asof_safe implementation:
        # It uses pd.merge_asof(direction='backward').
        # This should populate every row in price_data that has at least one prior fundamental record.
        
        # Let's verify the column names.
        # The input `fundamental_data` has columns like 'balance_sheet__totalAssets', 'income_statement__netIncome'.
        # But the feature computation methods look for 'pe_ratio', 'roe', 'revenue', etc.
        # THESE COLUMNS ARE MISSING in the input `fundamental_data`!
        # We need to MAP the raw EODHD columns to the internal feature names.
        
        # Mapping Raw EODHD -> Internal Feature Names
        # Coalesce total_assets from multiple possible columns
        # Prioritize existing total_assets, then balance_sheet__totalAssets, then sum of parts
        # We use combine_first to fill NaNs in the primary column with values from the backups
        if 'total_assets' not in df.columns:
            df['total_assets'] = np.nan
            
        if 'balance_sheet__totalAssets' in df.columns:
            df['total_assets'] = df['total_assets'].combine_first(df['balance_sheet__totalAssets'])
            
        if 'balance_sheet__nonCurrentAssetsTotal' in df.columns and 'balance_sheet__totalCurrentAssets' in df.columns:
            calculated_assets = df['balance_sheet__nonCurrentAssetsTotal'].fillna(0) + df['balance_sheet__totalCurrentAssets'].fillna(0)
            # Only use calculated if it's non-zero
            calculated_assets = calculated_assets.replace(0, np.nan)
            df['total_assets'] = df['total_assets'].combine_first(calculated_assets)
        
        # Revenue
        if 'income_statement__totalRevenue' in df.columns:
            if 'revenue' not in df.columns:
                 df['revenue'] = df['income_statement__totalRevenue']
            else:
                 df['revenue'] = df['revenue'].combine_first(df['income_statement__totalRevenue'])
        elif 'GeneralInfo__RevenueTTM' in df.columns: # Fallback
            if 'revenue' not in df.columns:
                df['revenue'] = df['GeneralInfo__RevenueTTM']
            else:
                df['revenue'] = df['revenue'].combine_first(df['GeneralInfo__RevenueTTM'])
            
        # Net Income
        if 'income_statement__netIncome' in df.columns:
            if 'net_income' not in df.columns:
                df['net_income'] = df['income_statement__netIncome']
            else:
                df['net_income'] = df['net_income'].combine_first(df['income_statement__netIncome'])
            
        # Operating Cash Flow (Critical for Accruals/Quality)
        if 'cash_flow__totalCashFromOperatingActivities' in df.columns:
            if 'operating_cash_flow' not in df.columns:
                df['operating_cash_flow'] = df['cash_flow__totalCashFromOperatingActivities']
            else:
                df['operating_cash_flow'] = df['operating_cash_flow'].combine_first(df['cash_flow__totalCashFromOperatingActivities'])
        elif 'cash_flow__operatingCashFlow' in df.columns:
             if 'operating_cash_flow' not in df.columns:
                 df['operating_cash_flow'] = df['cash_flow__operatingCashFlow']
             else:
                 df['operating_cash_flow'] = df['operating_cash_flow'].combine_first(df['cash_flow__operatingCashFlow'])
             
        # Accruals = (Net Income - Operating Cash Flow) / Total Assets
        # Lower is better (less manipulation)
        if 'accruals' not in df.columns and 'net_income' in df.columns and 'operating_cash_flow' in df.columns and 'total_assets' in df.columns:
            df['accruals'] = (df['net_income'] - df['operating_cash_flow']) / df['total_assets'].replace(0, np.nan)
            
        # Cash Flow to Net Income (Conversion Ratio)
        if 'cf_to_ni' not in df.columns and 'net_income' in df.columns and 'operating_cash_flow' in df.columns:
            df['cf_to_ni'] = df['operating_cash_flow'] / df['net_income'].replace(0, np.nan)

        # EPS (Calculated if missing)
        if 'eps' not in df.columns and 'net_income' in df.columns and 'balance_sheet__commonStockSharesOutstanding' in df.columns:
            df['eps'] = df['net_income'] / df['balance_sheet__commonStockSharesOutstanding'].replace(0, np.nan)
            
        # Equity
        if 'balance_sheet__totalStockholderEquity' in df.columns:
            df['total_equity'] = df['balance_sheet__totalStockholderEquity']
            
        # Debt
        if 'balance_sheet__shortLongTermDebtTotal' in df.columns:
            df['total_debt'] = df['balance_sheet__shortLongTermDebtTotal']
        elif 'balance_sheet__shortTermDebt' in df.columns and 'balance_sheet__longTermDebt' in df.columns:
            df['total_debt'] = df['balance_sheet__shortTermDebt'].fillna(0) + df['balance_sheet__longTermDebt'].fillna(0)
            
        # Cash
        if 'balance_sheet__cashAndEquivalents' in df.columns:
            df['cash_and_equivalents'] = df['balance_sheet__cashAndEquivalents']
            
        # Current Assets/Liabilities
        if 'balance_sheet__totalCurrentAssets' in df.columns:
            df['total_current_assets'] = df['balance_sheet__totalCurrentAssets']
        if 'balance_sheet__totalCurrentLiabilities' in df.columns:
            df['total_current_liabilities'] = df['balance_sheet__totalCurrentLiabilities']
            
        # Inventory
        if 'balance_sheet__inventory' in df.columns:
            df['inventory'] = df['balance_sheet__inventory']
            
        # Operating Income / EBIT
        if 'income_statement__operatingIncome' in df.columns:
            df['operating_income'] = df['income_statement__operatingIncome']
        elif 'income_statement__ebit' in df.columns:
            df['operating_income'] = df['income_statement__ebit']
            
        # Compute derived metrics if missing
        
        # ROE = Net Income / Total Equity
        if 'roe' not in df.columns and 'net_income' in df.columns and 'total_equity' in df.columns:
            df['roe'] = df['net_income'] / df['total_equity'].replace(0, np.nan)
            
        # ROA = Net Income / Total Assets
        if 'roa' not in df.columns and 'net_income' in df.columns and 'total_assets' in df.columns:
            df['roa'] = df['net_income'] / df['total_assets'].replace(0, np.nan)
            
        # Margins
        if 'revenue' in df.columns:
            if 'gross_margin' not in df.columns and 'income_statement__grossProfit' in df.columns:
                df['gross_margin'] = df['income_statement__grossProfit'] / df['revenue'].replace(0, np.nan)
            
            if 'operating_margin' not in df.columns and 'operating_income' in df.columns:
                df['operating_margin'] = df['operating_income'] / df['revenue'].replace(0, np.nan)
                
            if 'net_margin' not in df.columns and 'net_income' in df.columns:
                df['net_margin'] = df['net_income'] / df['revenue'].replace(0, np.nan)
        
        # Leverage
        if 'debt_to_equity' not in df.columns and 'total_debt' in df.columns and 'total_equity' in df.columns:
            df['debt_to_equity'] = df['total_debt'] / df['total_equity'].replace(0, np.nan)
            
        # Liquidity
        if 'current_ratio' not in df.columns and 'total_current_assets' in df.columns and 'total_current_liabilities' in df.columns:
            df['current_ratio'] = df['total_current_assets'] / df['total_current_liabilities'].replace(0, np.nan)
            
        if 'quick_ratio' not in df.columns and 'total_current_assets' in df.columns and 'inventory' in df.columns and 'total_current_liabilities' in df.columns:
            df['quick_ratio'] = (df['total_current_assets'] - df['inventory']) / df['total_current_liabilities'].replace(0, np.nan)
            
        # Cash to Debt
        if 'cash_to_debt' not in df.columns and 'cash_and_equivalents' in df.columns and 'total_debt' in df.columns:
            df['cash_to_debt'] = df['cash_and_equivalents'] / df['total_debt'].replace(0, np.nan)
            
        # Valuation (P/E, P/B, P/S)
        # Note: P/E is computed in _valuation_features using 'eps' and 'close'
        if 'pb_ratio' not in df.columns and 'close' in df.columns and 'total_equity' in df.columns and 'balance_sheet__commonStockSharesOutstanding' in df.columns:
            book_value_per_share = df['total_equity'] / df['balance_sheet__commonStockSharesOutstanding'].replace(0, np.nan)
            df['pb_ratio'] = df['close'] / book_value_per_share.replace(0, np.nan)
            
        if 'ps_ratio' not in df.columns and 'close' in df.columns and 'revenue' in df.columns and 'balance_sheet__commonStockSharesOutstanding' in df.columns:
            revenue_per_share = df['revenue'] / df['balance_sheet__commonStockSharesOutstanding'].replace(0, np.nan)
            df['ps_ratio'] = df['close'] / revenue_per_share.replace(0, np.nan)

        # Compute feature blocks
        features = pd.DataFrame(index=df.index)
        
        # Valuation features
        val_features = self._valuation_features(df)
        for col in val_features.columns:
            features[col] = val_features[col]
        
        # Profitability features
        prof_features = self._profitability_features(df)
        for col in prof_features.columns:
            features[col] = prof_features[col]
        
        # Growth features
        growth_features = self._growth_features(df)
        for col in growth_features.columns:
            features[col] = growth_features[col]
        
        # Balance sheet features
        bs_features = self._balance_sheet_features(df)
        for col in bs_features.columns:
            features[col] = bs_features[col]
        
        # Quality features
        quality_features = self._quality_features(df)
        for col in quality_features.columns:
            features[col] = quality_features[col]
        
        return features
    
    def _valuation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valuation features - highest ROI category.
        
        Features:
        - Raw multiples: pe_ratio, pb_ratio, ps_ratio
        - Sector-relative: pe_sector_rel, pb_sector_rel, ps_sector_rel
        - Sector z-scores: pe_sector_z, pb_sector_z
        - Valuation composite: cheap_signal
        """
        features = pd.DataFrame(index=df.index)
        
        # Raw valuation multiples (handle missing fields gracefully)
        # Compute P/E from price and EPS if available (preferred for point-in-time)
        if 'eps' in df.columns and 'close' in df.columns:
            # Annualize quarterly EPS (simple approximation: eps * 4)
            # Better: TTM EPS (requires rolling sum, but here we just have the merged row)
            # We'll use eps * 4 as a proxy for now
            annualized_eps = df['eps'] * 4
            features['pe_ratio'] = df['close'] / annualized_eps.replace(0, np.nan)
        else:
            features['pe_ratio'] = df.get('pe_ratio', pd.Series(index=df.index, dtype=float))

        features['pb_ratio'] = df.get('pb_ratio', pd.Series(index=df.index, dtype=float))
        features['ps_ratio'] = df.get('ps_ratio', pd.Series(index=df.index, dtype=float))
        
        # Inverse multiples (E/P, B/P, S/P) - sometimes more linear
        if 'pe_ratio' in df.columns:
            features['ep_ratio'] = 1.0 / df['pe_ratio'].replace(0, np.nan)
        if 'pb_ratio' in df.columns:
            features['bp_ratio'] = 1.0 / df['pb_ratio'].replace(0, np.nan)
        if 'ps_ratio' in df.columns:
            features['sp_ratio'] = 1.0 / df['ps_ratio'].replace(0, np.nan)
        
        # Sector-relative valuation (core signal)
        if 'sector' in df.columns:
            # Relative to sector mean
            if 'pe_ratio' in df.columns:
                features['pe_sector_rel'] = self._sector_relative(
                    df, 'pe_ratio', method='relative_mean'
                )
            if 'pb_ratio' in df.columns:
                features['pb_sector_rel'] = self._sector_relative(
                    df, 'pb_ratio', method='relative_mean'
                )
            if 'ps_ratio' in df.columns:
                features['ps_sector_rel'] = self._sector_relative(
                    df, 'ps_ratio', method='relative_mean'
                )
            
            # Sector z-scores (normalized cross-section)
            if 'pe_ratio' in df.columns:
                features['pe_sector_z'] = self._sector_relative(
                    df, 'pe_ratio', method='zscore'
                )
            if 'pb_ratio' in df.columns:
                features['pb_sector_z'] = self._sector_relative(
                    df, 'pb_ratio', method='zscore'
                )
            if 'ps_ratio' in df.columns:
                features['ps_sector_z'] = self._sector_relative(
                    df, 'ps_ratio', method='zscore'
                )
            
            # Sector percentile ranks
            if 'pe_ratio' in df.columns:
                features['pe_sector_rank'] = self._sector_relative(
                    df, 'pe_ratio', method='rank_pct'
                )
            if 'pb_ratio' in df.columns:
                features['pb_sector_rank'] = self._sector_relative(
                    df, 'pb_ratio', method='rank_pct'
                )
        
        # Composite "cheap" signal
        # Low P/E + Low P/B = value stock
        pe_z = features.get('pe_sector_z', pd.Series(0, index=df.index))
        pb_z = features.get('pb_sector_z', pd.Series(0, index=df.index))
        features['value_composite'] = (-pe_z.fillna(0) + -pb_z.fillna(0)) / 2.0
        
        # PEG ratio (P/E to growth) - if we have growth data
        if 'peg_ratio' in df.columns:
            features['peg_ratio'] = df['peg_ratio']
            features['peg_sector_z'] = self._sector_relative(
                df, 'peg_ratio', method='zscore'
            )
        
        return features
    
    def _profitability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Profitability and quality features.
        
        Features:
        - ROE, ROA
        - Margins (gross, operating, net)
        - Sector-relative profitability
        - Profitability trends
        """
        features = pd.DataFrame(index=df.index)
        
        # Raw profitability metrics (handle missing fields)
        features['roe'] = df.get('roe', pd.Series(index=df.index, dtype=float))
        features['roa'] = df.get('roa', pd.Series(index=df.index, dtype=float))
        features['gross_margin'] = df.get('gross_margin', pd.Series(index=df.index, dtype=float))
        features['operating_margin'] = df.get('operating_margin', pd.Series(index=df.index, dtype=float))
        features['net_margin'] = df.get('net_margin', pd.Series(index=df.index, dtype=float))
        
        # Sector-relative profitability
        if 'sector' in df.columns:
            if 'roe' in df.columns:
                features['roe_sector_z'] = self._sector_relative(
                    df, 'roe', method='zscore'
                )
            if 'roa' in df.columns:
                features['roa_sector_z'] = self._sector_relative(
                    df, 'roa', method='zscore'
                )
            if 'net_margin' in df.columns:
                features['net_margin_sector_z'] = self._sector_relative(
                    df, 'net_margin', method='zscore'
                )
            
            if 'roe' in df.columns:
                features['roe_sector_rank'] = self._sector_relative(
                    df, 'roe', method='rank_pct'
                )
        
        # Profitability composite
        roe_z = features.get('roe_sector_z', pd.Series(0, index=df.index))
        margin_z = features.get('net_margin_sector_z', pd.Series(0, index=df.index))
        features['profitability_composite'] = (roe_z.fillna(0) + margin_z.fillna(0)) / 2.0
        
        # Profitability trends (requires time series)
        if 'ticker' in df.columns:
            ticker_group = df.groupby('ticker')
            
            # 4-quarter ROE trend (if available)
            if 'roe' in df.columns:
                features['roe_trend_4q'] = ticker_group['roe'].transform(
                    lambda x: self._linear_trend(x, periods=4)
                )
            
            # Margin expansion (if available)
            if 'net_margin' in df.columns:
                features['margin_expansion_4q'] = ticker_group['net_margin'].transform(
                    lambda x: x.diff(4)
                )
        
        return features
    
    def _growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Growth features - revenue and earnings growth.
        
        Features:
        - YoY revenue growth
        - YoY earnings growth
        - Growth acceleration
        - Sector-relative growth
        """
        features = pd.DataFrame(index=df.index)
        
        # If we don't have revenue/earnings, can't compute growth
        if 'revenue' not in df.columns or 'ticker' not in df.columns:
            return features
        
        ticker_group = df.groupby('ticker')
        
        # Year-over-year growth (4 quarters)
        features['revenue_growth_yoy'] = ticker_group['revenue'].transform(
            lambda x: x.pct_change(4)
        )
        
        if 'net_income' in df.columns:
            features['earnings_growth_yoy'] = ticker_group['net_income'].transform(
                lambda x: x.pct_change(4)
            )
        
        if 'eps' in df.columns:
            features['eps_growth_yoy'] = ticker_group['eps'].transform(
                lambda x: x.pct_change(4)
            )
        
        # Growth acceleration (change in growth rate)
        features['revenue_acceleration'] = ticker_group['revenue'].transform(
            lambda x: x.pct_change(4).diff()
        )
        
        # Sector-relative growth
        if 'sector' in df.columns:
            features['revenue_growth_sector_z'] = self._sector_relative(
                df, 'revenue_growth_yoy', method='zscore'
            )
            features['earnings_growth_sector_z'] = self._sector_relative(
                df, 'earnings_growth_yoy', method='zscore'
            )
        
        # Growth quality (consistent vs volatile)
        features['revenue_growth_stability'] = ticker_group['revenue'].transform(
            lambda x: x.pct_change(4).rolling(8).std()
        )
        
        return features
    
    def _balance_sheet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance sheet strength features.
        
        Features:
        - Debt ratios
        - Liquidity ratios
        - Cash position
        - Sector-relative leverage
        """
        features = pd.DataFrame(index=df.index)
        
        # Leverage (handle missing fields)
        features['debt_to_equity'] = df.get('debt_to_equity', pd.Series(index=df.index, dtype=float))
        features['current_ratio'] = df.get('current_ratio', pd.Series(index=df.index, dtype=float))
        features['quick_ratio'] = df.get('quick_ratio', pd.Series(index=df.index, dtype=float))
        features['cash_to_debt'] = df.get('cash_to_debt', pd.Series(index=df.index, dtype=float))
        
        # Sector-relative leverage
        if 'sector' in df.columns:
            if 'debt_to_equity' in df.columns:
                features['debt_to_equity_sector_z'] = self._sector_relative(
                    df, 'debt_to_equity', method='zscore'
                )
            if 'current_ratio' in df.columns:
                features['current_ratio_sector_z'] = self._sector_relative(
                    df, 'current_ratio', method='zscore'
                )
        
        # Financial strength composite
        # Low debt + high liquidity = strong balance sheet
        debt_z = features.get('debt_to_equity_sector_z', pd.Series(0, index=df.index))
        curr_z = features.get('current_ratio_sector_z', pd.Series(0, index=df.index))
        features['financial_strength'] = (-debt_z.fillna(0) + curr_z.fillna(0)) / 2.0
        
        return features
    
    def _quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Earnings quality features.
        
        Features:
        - Accruals ratio
        - Cash flow to net income
        - Operating cash flow quality
        """
        features = pd.DataFrame(index=df.index)
        
        # Accruals (lower is better quality) - handle missing
        features['accruals'] = df.get('accruals', pd.Series(index=df.index, dtype=float))
        
        # Cash conversion
        features['cf_to_ni'] = df.get('cf_to_ni', pd.Series(index=df.index, dtype=float))
        
        # Sector-relative quality
        if 'sector' in df.columns:
            # Low accruals is high quality, so negate the z-score
            features['accruals_sector_z'] = self._sector_relative(
                df, 'accruals', method='zscore'
            )
            features['quality_score'] = -features['accruals_sector_z'].fillna(0)
        
        return features
    
    def _sector_relative(
        self, 
        df: pd.DataFrame, 
        column: str, 
        method: str = 'relative_mean'
    ) -> pd.Series:
        """
        Compute sector-relative transformation.
        
        Args:
            df: DataFrame with 'sector', 'date', and column
            column: Column to transform
            method: 'relative_mean', 'zscore', or 'rank_pct'
            
        Returns:
            Series with sector-relative values
        """
        if column not in df.columns or df[column].isna().all():
            return pd.Series(np.nan, index=df.index)
        
        # Group by date and sector for cross-sectional transformation
        grouped = df.groupby(['date', 'sector'])[column]
        
        if method == 'relative_mean':
            # (value / sector_mean) - 1
            sector_mean = grouped.transform('mean')
            rel = (df[column] / sector_mean.replace(0, np.nan)) - 1.0
            # If sector_mean is NaN, rel is NaN.
            # If df[column] is NaN, rel is NaN.
            # We might want to fill 0.0 if sector_mean is missing but value exists?
            # For now, let's keep NaNs to be safe.
            return rel
        
        elif method == 'zscore':
            # (value - sector_mean) / sector_std
            sector_mean = grouped.transform('mean')
            sector_std = grouped.transform('std')
            z = (df[column] - sector_mean) / sector_std.replace(0, np.nan)
            # Only fill NaN if the original value was NOT NaN (i.e. sector stats missing)
            # If original value was NaN, keep it NaN
            mask = df[column].notna()
            z[mask] = z[mask].fillna(0.0)
            return z
        
        elif method == 'rank_pct':
            # Percentile rank within sector
            return grouped.rank(pct=True)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _linear_trend(self, series: pd.Series, periods: int = 4) -> pd.Series:
        """Compute linear trend over rolling window."""
        def trend(x):
            if len(x) < 2 or x.isna().any():
                return np.nan
            y = np.arange(len(x))
            slope, _ = np.polyfit(y, x, 1)
            return slope
        
        return series.rolling(periods).apply(trend, raw=False)
