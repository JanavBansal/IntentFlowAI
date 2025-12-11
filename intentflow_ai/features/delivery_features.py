"""
Delivery Features for Flow Detective Agent.

These features detect institutional accumulation patterns from NSE delivery data.
This is IntentFlow V4.5's SECRET WEAPON - institutional investors leave footprints
in delivery volume that precede price moves.

Features:
- Delivery percentage momentum (3d, 7d, 14d changes)
- Delivery quantity z-scores (unusual volume detector)
- Large order flags (98th percentile spikes)
- Accumulation/Distribution line (Chaikin-style)
- Delivery to total volume ratio trends
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def compute_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delivery-based features for Flow Detective agent.
    
    Input DataFrame must have columns:
    - delivery_qty: Absolute delivery quantity
    - delivery_pct OR delivery_ratio: Delivery as % of total volume (0-100 or 0-1)
    - volume: Total traded volume
    - close: Close price (for value calculations)
    - high, low: For money flow calculations
    
    Output: DataFrame with 15+ new delivery-focused features appended.
    """
    result = df.copy()
    
    # Normalize delivery_pct (handle both 0-100 and 0-1 formats)
    if 'delivery_pct' in result.columns:
        delivery_pct = result['delivery_pct']
        if delivery_pct.max() > 1.5:  # Likely 0-100 format
            result['delivery_ratio'] = delivery_pct / 100.0
        else:
            result['delivery_ratio'] = delivery_pct
    elif 'delivery_ratio' in result.columns:
        pass  # Already have delivery_ratio
    else:
        logger.warning("No delivery_pct or delivery_ratio column found. Returning features with zeros.")
        _add_zero_delivery_features(result)
        return result
    
    # Ensure delivery_qty exists
    if 'delivery_qty' not in result.columns:
        if 'deliveryqty' in result.columns:
            result['delivery_qty'] = result['deliveryqty']
        elif 'volume' in result.columns and 'delivery_ratio' in result.columns:
            result['delivery_qty'] = result['volume'] * result['delivery_ratio']
        else:
            result['delivery_qty'] = 0
    
    # ==========================================================================
    # 1. Delivery Percentage Momentum
    # ==========================================================================
    result['deliv_pct_3d_change'] = result['delivery_ratio'].pct_change(3)
    result['deliv_pct_7d_change'] = result['delivery_ratio'].pct_change(7)
    result['deliv_pct_14d_change'] = result['delivery_ratio'].pct_change(14)
    
    # ==========================================================================
    # 2. Delivery Volume Z-Scores (detects institutional buying spikes)
    # ==========================================================================
    deliv_mean_20 = result['delivery_qty'].rolling(20, min_periods=5).mean()
    deliv_std_20 = result['delivery_qty'].rolling(20, min_periods=5).std()
    
    result['deliv_qty_zscore'] = (result['delivery_qty'] - deliv_mean_20) / deliv_std_20.replace(0, np.nan)
    result['deliv_qty_zscore'] = result['deliv_qty_zscore'].fillna(0)
    
    # ==========================================================================
    # 3. Large Order Flag (98th percentile = institutional activity)
    # ==========================================================================
    # Rolling percentile to make it adaptive
    def rolling_percentile_98(s: pd.Series, window: int = 60) -> pd.Series:
        return s.rolling(window, min_periods=20).apply(lambda x: 1 if x.iloc[-1] > np.percentile(x, 98) else 0)
    
    result['large_delivery_flag'] = (result['deliv_qty_zscore'] > 2.0).astype(int)
    
    # ==========================================================================
    # 4. Accumulation/Distribution Line (Chaikin-style with delivery)
    # ==========================================================================
    if all(col in result.columns for col in ['high', 'low', 'close']):
        price_range = result['high'] - result['low']
        price_range = price_range.replace(0, np.nan)
        
        # Money flow multiplier: [-1, 1]
        result['money_flow_mult'] = (
            (result['close'] - result['low']) - (result['high'] - result['close'])
        ) / price_range
        result['money_flow_mult'] = result['money_flow_mult'].fillna(0)
        
        # Money flow volume (using delivery instead of total volume)
        result['money_flow_volume'] = result['money_flow_mult'] * result['delivery_qty']
        
        # Cumulative A/D line
        result['accum_distrib'] = result['money_flow_volume'].cumsum()
        
        # A/D momentum
        result['accum_distrib_5d'] = result['accum_distrib'].diff(5)
        result['accum_distrib_10d'] = result['accum_distrib'].diff(10)
    else:
        result['money_flow_mult'] = 0.0
        result['money_flow_volume'] = 0.0
        result['accum_distrib'] = 0.0
        result['accum_distrib_5d'] = 0.0
        result['accum_distrib_10d'] = 0.0
    
    # ==========================================================================
    # 5. Delivery to Volume Ratio Trends
    # ==========================================================================
    result['deliv_ratio_ma_5'] = result['delivery_ratio'].rolling(5, min_periods=2).mean()
    result['deliv_ratio_ma_20'] = result['delivery_ratio'].rolling(20, min_periods=5).mean()
    result['deliv_ratio_ma_50'] = result['delivery_ratio'].rolling(50, min_periods=10).mean()
    
    # Short vs long delivery ratio (rising = accumulation)
    result['deliv_ratio_trend'] = result['deliv_ratio_ma_5'] / result['deliv_ratio_ma_20'].replace(0, np.nan)
    result['deliv_ratio_trend'] = result['deliv_ratio_trend'].fillna(1.0)
    
    # ==========================================================================
    # 6. Delivery Value (Rs amount - bigger values = more conviction)
    # ==========================================================================
    if 'close' in result.columns:
        result['deliv_value'] = result['delivery_qty'] * result['close']
        deliv_value_mean_20 = result['deliv_value'].rolling(20, min_periods=5).mean()
        result['deliv_value_spike'] = result['deliv_value'] / deliv_value_mean_20.replace(0, np.nan)
        result['deliv_value_spike'] = result['deliv_value_spike'].fillna(1.0)
    else:
        result['deliv_value'] = 0.0
        result['deliv_value_spike'] = 1.0
    
    # ==========================================================================
    # 7. Delivery vs Price Return Correlation (rolling)
    # ==========================================================================
    if 'close' in result.columns:
        returns = result['close'].pct_change()
        result['deliv_price_corr_10'] = (
            result['delivery_ratio']
            .rolling(10, min_periods=5)
            .corr(returns)
            .fillna(0)
        )
    else:
        result['deliv_price_corr_10'] = 0.0
    
    # Clean up infinities
    for col in result.columns:
        if result[col].dtype in [np.float64, np.float32]:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result


def _add_zero_delivery_features(df: pd.DataFrame) -> None:
    """Add delivery features with zero values when delivery data unavailable."""
    zero_cols = [
        'delivery_ratio', 'delivery_qty',
        'deliv_pct_3d_change', 'deliv_pct_7d_change', 'deliv_pct_14d_change',
        'deliv_qty_zscore', 'large_delivery_flag',
        'money_flow_mult', 'money_flow_volume', 'accum_distrib',
        'accum_distrib_5d', 'accum_distrib_10d',
        'deliv_ratio_ma_5', 'deliv_ratio_ma_20', 'deliv_ratio_ma_50',
        'deliv_ratio_trend', 'deliv_value', 'deliv_value_spike',
        'deliv_price_corr_10'
    ]
    for col in zero_cols:
        if col not in df.columns:
            df[col] = 0.0


def get_delivery_feature_names() -> List[str]:
    """Return list of all delivery feature names for agent configuration."""
    return [
        # Momentum
        'deliv_pct_3d_change',
        'deliv_pct_7d_change',
        'deliv_pct_14d_change',
        
        # Z-scores
        'deliv_qty_zscore',
        'large_delivery_flag',
        
        # Accumulation/Distribution
        'money_flow_mult',
        'money_flow_volume',
        'accum_distrib_5d',
        'accum_distrib_10d',
        
        # Ratios
        'deliv_ratio_ma_5',
        'deliv_ratio_ma_20',
        'deliv_ratio_trend',
        
        # Value
        'deliv_value_spike',
        
        # Correlation
        'deliv_price_corr_10',
    ]
