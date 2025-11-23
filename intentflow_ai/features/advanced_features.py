"""
Advanced feature blocks for improved model performance.
"""
import pandas as pd
import numpy as np


def sector_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sector momentum features: how is this ticker performing vs its sector?
    """
    if 'sector' not in df.columns or df['sector'].isna().all():
        return pd.DataFrame(index=df.index)
    
    features = pd.DataFrame(index=df.index)
    
    # Group by sector and date
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate sector average returns
    for period in [5, 20, 60]:
        df[f'ret_{period}d_temp'] = df.groupby('ticker')['close'].pct_change(period)
        
        # Sector mean return
        sector_mean = df.groupby(['date', 'sector'])[f'ret_{period}d_temp'].transform('mean')
        features[f'sector_momentum_{period}d'] = df[f'ret_{period}d_temp'] - sector_mean
        
        # Sector return spread (std)
        sector_std = df.groupby(['date', 'sector'])[f'ret_{period}d_temp'].transform('std')
        features[f'sector_dispersion_{period}d'] = sector_std
    
    # Sector rank (percentile within sector)
    df['close_temp'] = df['close']
    features['sector_rank'] = df.groupby(['date', 'sector'])['close_temp'].rank(pct=True)
    
    return features


def earnings_metrics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Earnings-based features: surprises, growth, consistency.
    Requires fundamental data columns.
    """
    features = pd.DataFrame(index=df.index)
    
    # Check if we have fundamental columns
    fund_cols = ['revenue', 'net_income', 'ebitda', 'eps']
    available_cols = [col for col in fund_cols if col in df.columns]
    
    if len(available_cols) == 0:
        return features
    
    df = df.copy()
    
    for col in available_cols:
        if col in df.columns:
            # Year-over-year growth (4 quarters)
            df[f'{col}_yoy'] = df.groupby('ticker')[col].pct_change(4)
            features[f'{col}_growth_yoy'] = df[f'{col}_yoy']
            
            # Quarter-over-quarter growth
            df[f'{col}_qoq'] = df.groupby('ticker')[col].pct_change(1)
            features[f'{col}_growth_qoq'] = df[f'{col}_qoq']
            
            # Growth acceleration (change in growth rate)
            features[f'{col}_growth_accel'] = df.groupby('ticker')[f'{col}_yoy'].diff()
            
            # Earnings consistency (std of last 4 quarters growth)
            features[f'{col}_consistency'] = df.groupby('ticker')[f'{col}_qoq'].rolling(4, min_periods=2).std().reset_index(level=0, drop=True)
    
    # Earnings surprise proxy (deviation from trend)
    if 'net_income' in df.columns:
        df['ni_ma4'] = df.groupby('ticker')['net_income'].rolling(4, min_periods=2).mean().reset_index(level=0, drop=True)
        features['earnings_surprise_proxy'] = (df['net_income'] - df['ni_ma4']) / (df['ni_ma4'].abs() + 1e-6)
    
    return features


def quality_scores_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quality score features: profitability, efficiency, financial health.
    """
    features = pd.DataFrame(index=df.index)
    
    df = df.copy()
    
    # ROE (Return on Equity)
    if 'net_income' in df.columns and 'total_equity' in df.columns:
        df['roe'] = df['net_income'] / (df['total_equity'].abs() + 1e-6)
        features['roe'] = df['roe']
        features['roe_stability'] = df.groupby('ticker')['roe'].rolling(4, min_periods=2).std().reset_index(level=0, drop=True)
    
    # ROA (Return on Assets)
    if 'net_income' in df.columns and 'total_assets' in df.columns:
        df['ro'] = df['net_income'] / (df['total_assets'].abs() + 1e-6)
        features['roa'] = df['roa']
    
    # Profit margins
    if 'net_income' in df.columns and 'revenue' in df.columns:
        df['profit_margin'] = df['net_income'] / (df['revenue'].abs() + 1e-6)
        features['profit_margin'] = df['profit_margin']
        features['margin_trend'] = df.groupby('ticker')['profit_margin'].diff()
    
    if 'ebitda' in df.columns and 'revenue' in df.columns:
        features['ebitda_margin'] = df['ebitda'] / (df['revenue'].abs() + 1e-6)
    
    # Asset efficiency
    if 'revenue' in df.columns and 'total_assets' in df.columns:
        features['asset_turnover'] = df['revenue'] / (df['total_assets'].abs() + 1e-6)
    
    # Cash conversion (OCF / Net Income)
    if 'operating_cash_flow' in df.columns and 'net_income' in df.columns:
        features['cash_conversion'] = df['operating_cash_flow'] / (df['net_income'].abs() + 1e-6)
    
    # Leverage (Debt to Equity)
    if 'total_liabilities' in df.columns and 'total_equity' in df.columns:
        features['debt_to_equity'] = df['total_liabilities'] / (df['total_equity'].abs() + 1e-6)
    
    # Piotroski F-Score components (simplified)
    if 'net_income' in df.columns:
        features['profitability_flag'] = (df['net_income'] > 0).astype(int)
    
    if 'operating_cash_flow' in df.columns:
        features['cash_flow_flag'] = (df['operating_cash_flow'] > 0).astype(int)
    
    return features
