"""Time-based dataset splits."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def time_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    valid_start: str,
    test_start: str,
    embargo_days: int = 0,
    horizon_days: int = 0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return boolean masks for train/valid/test partitions with optional purging.

    An embargo window can be supplied to drop observations that have overlapping
    label horizons with downstream splits (to mitigate leakage). The effective
    purge window defaults to the prediction horizon when not explicitly set.
    """

    if valid_start >= test_start:
        raise ValueError("valid_start must be earlier than test_start.")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    valid_start_dt = pd.to_datetime(valid_start)
    test_start_dt = pd.to_datetime(test_start)
    buffer_days = max(int(embargo_days or 0), int(horizon_days or 0), 0)
    buffer = pd.Timedelta(days=buffer_days)

    train_cutoff = valid_start_dt - buffer
    valid_cutoff = test_start_dt - buffer

    train_mask = dates < train_cutoff
    valid_mask = (dates >= valid_start_dt) & (dates < valid_cutoff)
    test_mask = dates >= test_start_dt

    if not train_mask.any():
        raise ValueError("Training range is empty; adjust valid_start.")
    if not valid_mask.any():
        raise ValueError("Validation range is empty; adjust valid_start/test_start.")
    if not test_mask.any():
        raise ValueError("Test range is empty; adjust test_start.")

    return train_mask, valid_mask, test_mask


def purged_time_series_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    n_splits: int = 3,
    embargo_days: int = 0,
    horizon_days: int = 0,
) -> List[Tuple[pd.Series, pd.Series]]:
    """Generate (train_mask, test_mask) tuples with purging/embargo around each fold."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for purged CV.")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    unique_dates = dates.dropna().sort_values().unique()
    if len(unique_dates) < n_splits:
        raise ValueError("Not enough unique dates to create the requested folds.")

    fold_size = len(unique_dates) // n_splits
    buffer = pd.Timedelta(days=max(int(embargo_days), int(horizon_days), 0))
    splits: List[Tuple[pd.Series, pd.Series]] = []

    for fold_idx in range(n_splits):
        start_idx = fold_idx * fold_size
        end_idx = len(unique_dates) if fold_idx == n_splits - 1 else (fold_idx + 1) * fold_size
        valid_dates = unique_dates[start_idx:end_idx]
        if valid_dates.size == 0:
            continue
        start_date = valid_dates[0]
        end_date = valid_dates[-1]
        test_mask = (dates >= start_date) & (dates <= end_date)
        train_mask = dates < (start_date - buffer)
        splits.append((train_mask, test_mask))
    return splits
