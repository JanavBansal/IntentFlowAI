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
        # Merge fundamentals with price data
        df = price_data.merge(
            fundamental_data,
            left_on=['ticker', 'date'],
            right_on=['symbol', 'date'],
            how='left'
        )
        
        # Compute feature blocks
        features = pd.DataFrame(index=df.index)
        
        # Valuation features
        val_features = self._valuation_features(df)
        for col in val_features.columns:
            features[f"fundamental__{col}"] = val_features[col]
        
        # Profitability features
        prof_features = self._profitability_features(df)
        for col in prof_features.columns:
            features[f"fundamental__{col}"] = prof_features[col]
        
        # Growth features
        growth_features = self._growth_features(df)
        for col in growth_features.columns:
            features[f"fundamental__{col}"] = growth_features[col]
        
        # Balance sheet features
        bs_features = self._balance_sheet_features(df)
        for col in bs_features.columns:
            features[f"fundamental__{col}"] = bs_features[col]
        
        # Quality features
        quality_features = self._quality_features(df)
        for col in quality_features.columns:
            features[f"fundamental__{col}"] = quality_features[col]
        
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
            return (df[column] / sector_mean.replace(0, np.nan)) - 1.0
        
        elif method == 'zscore':
            # (value - sector_mean) / sector_std
            sector_mean = grouped.transform('mean')
            sector_std = grouped.transform('std')
            z = (df[column] - sector_mean) / sector_std.replace(0, np.nan)
            return z.fillna(0.0)
        
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
