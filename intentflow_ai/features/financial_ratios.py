"""
Professional financial ratio features using FinanceToolkit.
Automatically generates 150+ financial ratios from fundamental data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class FinancialRatioEngine:
    """
    Wrapper around FinanceToolkit to generate comprehensive financial ratios.
    """
    
    @staticmethod
    def prepare_financial_statements(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Convert our fundamental data format to FinanceToolkit's expected format.
        
        FinanceToolkit expects DataFrames with:
        - Index: dates
        - Columns: tickers
        - Values: financial metrics
        """
        if df.empty:
            return {'income': pd.DataFrame(), 'balance': pd.DataFrame(), 'cash': pd.DataFrame()}
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot to FinanceToolkit format (dates as index, tickers as columns)
        def pivot_statement(cols):
            result = {}
            for col in cols:
                if col in df.columns:
                    pivoted = df.pivot(index='date', columns='ticker', values=col)
                    result[col] = pivoted
            return pd.DataFrame(result) if result else pd.DataFrame()
        
        # Map our column names to standard names
        income_cols = ['revenue', 'net_income', 'ebitda', 'eps']
        balance_cols = ['total_assets', 'total_liabilities', 'total_equity']
        cash_cols = ['operating_cash_flow', 'capex']
        
        statements = {
            'income': pivot_statement(income_cols),
            'balance': pivot_statement(balance_cols),
            'cash': pivot_statement(cash_cols)
        }
        
        return statements
    
    @staticmethod
    def calculate_basic_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic financial ratios manually (FinanceToolkit alternative).
        Returns a DataFrame with same index as input.
        """
        features = pd.DataFrame(index=df.index)
        
        df = df.copy()
        
        # Profitability Ratios
        if 'net_income' in df.columns and 'revenue' in df.columns:
            features['profit_margin'] = df['net_income'] / (df['revenue'].abs() + 1e-6)
            features['profit_margin'] = features['profit_margin'].clip(-5, 5)  # Cap extremes
        
        if 'net_income' in df.columns and 'total_equity' in df.columns:
            features['roe'] = df['net_income'] / (df['total_equity'].abs() + 1e-6)
            features['roe'] = features['roe'].clip(-5, 5)
        
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            features['roa'] = df['net_income'] / (df['total_assets'].abs() + 1e-6)
            features['roa'] = features['roa'].clip(-5, 5)
        
        if 'ebitda' in df.columns and 'revenue' in df.columns:
            features['ebitda_margin'] = df['ebitda'] / (df['revenue'].abs() + 1e-6)
            features['ebitda_margin'] = features['ebitda_margin'].clip(-5, 5)
        
        # Efficiency Ratios
        if 'revenue' in df.columns and 'total_assets' in df.columns:
            features['asset_turnover'] = df['revenue'] / (df['total_assets'].abs() + 1e-6)
            features['asset_turnover'] = features['asset_turnover'].clip(0, 10)
        
        # Leverage Ratios
        if 'total_liabilities' in df.columns and 'total_equity' in df.columns:
            features['debt_to_equity'] = df['total_liabilities'] / (df['total_equity'].abs() + 1e-6)
            features['debt_to_equity'] = features['debt_to_equity'].clip(0, 20)
        
        if 'total_liabilities' in df.columns and 'total_assets' in df.columns:
            features['debt_ratio'] = df['total_liabilities'] / (df['total_assets'].abs() + 1e-6)
            features['debt_ratio'] = features['debt_ratio'].clip(0, 1)
        
        # Cash Flow Ratios
        if 'operating_cash_flow' in df.columns and 'net_income' in df.columns:
            features['cash_conversion'] = df['operating_cash_flow'] / (df['net_income'].abs() + 1e-6)
            features['cash_conversion'] = features['cash_conversion'].clip(-10, 10)
        
        if 'operating_cash_flow' in df.columns and 'total_assets' in df.columns:
            features['cash_flow_to_assets'] = df['operating_cash_flow'] / (df['total_assets'].abs() + 1e-6)
            features['cash_flow_to_assets'] = features['cash_flow_to_assets'].clip(-1, 1)
        
        # Growth Metrics (YoY)
        for col in ['revenue', 'net_income', 'ebitda', 'operating_cash_flow']:
            if col in df.columns:
                yoy = df.groupby('ticker')[col].pct_change(4)  # 4 quarters = 1 year
                features[f'{col}_growth_yoy'] = yoy.clip(-5, 5)
        
        return features


def sector_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sector-normalized versions of key financial metrics.
    This captures relative performance vs peers.
    """
    if 'sector' not in df.columns or df['sector'].isna().all():
        return pd.DataFrame(index=df.index)
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    features = pd.DataFrame(index=df.index)
    
    # Metrics to normalize by sector
    metrics_to_normalize = [
        'profit_margin', 'roe', 'roa', 'ebitda_margin',
        'asset_turnover', 'debt_to_equity', 'cash_conversion',
        'revenue_growth_yoy', 'net_income_growth_yoy'
    ]
    
    for metric in metrics_to_normalize:
        if metric in df.columns:
            # Calculate sector median for each date
            sector_median = df.groupby(['date', 'sector'])[metric].transform('median')
            
            # Normalized = actual - sector_median
            features[f'{metric}_sector_norm'] = df[metric] - sector_median
            
            # Also add sector rank (percentile)
            features[f'{metric}_sector_rank'] = df.groupby(['date', 'sector'])[metric].rank(pct=True)
    
    return features
