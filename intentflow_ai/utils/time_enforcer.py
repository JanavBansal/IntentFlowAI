"""
Time enforcement utilities for point-in-time correctness.

This module provides strict utilities for merging data to ensure no future information
leaks into the training set. It centralizes the logic for "as-of" merging and
reporting delay application.
"""

from typing import Optional, Union, List

import pandas as pd
import numpy as np


class TimeEnforcer:
    """Enforces point-in-time correctness for data merging."""
    
    @staticmethod
    def apply_reporting_delay(
        df: pd.DataFrame,
        report_date_col: str = "report_date",
        delay_days: int = 45,
        output_col: str = "available_date"
    ) -> pd.DataFrame:
        """
        Calculate when data becomes available based on reporting delay.
        
        Args:
            df: DataFrame containing report dates
            report_date_col: Column name for report date (e.g., quarter end)
            delay_days: Number of days to add for reporting delay
            output_col: Name of the new column for available date
            
        Returns:
            DataFrame with new available_date column
        """
        if df.empty or report_date_col not in df.columns:
            return df
        
        df = df.copy()
        df[report_date_col] = pd.to_datetime(df[report_date_col])
        df[output_col] = df[report_date_col] + pd.Timedelta(days=delay_days)
        return df

    @staticmethod
    def merge_asof_safe(
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_date_col: str,
        right_date_col: str,
        by: Union[str, List[str]],
        suffixes: tuple = ('', '_right'),
        tolerance: Optional[pd.Timedelta] = None
    ) -> pd.DataFrame:
        """
        Strict point-in-time merge using merge_asof.
        
        Ensures that for every row in 'left', we only match rows in 'right'
        where right[right_date_col] <= left[left_date_col].
        
        Args:
            left: Left DataFrame (e.g., price data) - the "target" timeline
            right: Right DataFrame (e.g., fundamental data) - the "source" data
            left_date_col: Date column in left DataFrame
            right_date_col: Date column in right DataFrame (must be available_date)
            by: Column(s) to group by (e.g., 'ticker')
            suffixes: Suffixes for overlapping columns
            tolerance: Optional tolerance for lookback
            
        Returns:
            Merged DataFrame
        """
        if left.empty or right.empty:
            return left
            
        # Ensure dates are datetime
        left = left.copy()
        right = right.copy()
        
        left[left_date_col] = pd.to_datetime(left[left_date_col])
        right[right_date_col] = pd.to_datetime(right[right_date_col])
        
        # Sort required for merge_asof
        left = left.sort_values(left_date_col)
        right = right.sort_values(right_date_col)
        
        # If 'by' is a list, merge_asof only supports single 'by' in older pandas,
        # but modern pandas supports list. We'll assume list is supported or 'by' is string.
        
        try:
            merged = pd.merge_asof(
                left,
                right,
                left_on=left_date_col,
                right_on=right_date_col,
                by=by,
                direction='backward',
                suffixes=suffixes,
                tolerance=tolerance
            )
            return merged
        except Exception as e:
            # Fallback for complex cases or errors: iterate by group
            # This is slower but safer
            print(f"Warning: merge_asof failed ({e}), falling back to iterative merge")
            
            if isinstance(by, str):
                by = [by]
                
            merged_dfs = []
            for group_keys, left_group in left.groupby(by):
                # Find matching right group
                # Construct query for right group
                right_group = right
                for col, val in zip(by, group_keys if isinstance(group_keys, tuple) else [group_keys]):
                    right_group = right_group[right_group[col] == val]
                
                if right_group.empty:
                    merged_dfs.append(left_group)
                    continue
                    
                # Perform merge_asof on the group
                m = pd.merge_asof(
                    left_group.sort_values(left_date_col),
                    right_group.sort_values(right_date_col),
                    left_on=left_date_col,
                    right_on=right_date_col,
                    direction='backward',
                    suffixes=suffixes,
                    tolerance=tolerance
                )
                merged_dfs.append(m)
            
            if not merged_dfs:
                return left
                
            return pd.concat(merged_dfs, ignore_index=True)
