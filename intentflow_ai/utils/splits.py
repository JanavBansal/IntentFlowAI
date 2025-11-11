"""Time-based dataset splits."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def time_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    valid_start: str,
    test_start: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return boolean masks for train/valid/test partitions."""

    if valid_start >= test_start:
        raise ValueError("valid_start must be earlier than test_start.")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    valid_start_dt = pd.to_datetime(valid_start)
    test_start_dt = pd.to_datetime(test_start)

    train_mask = dates < valid_start_dt
    valid_mask = (dates >= valid_start_dt) & (dates < test_start_dt)
    test_mask = dates >= test_start_dt

    if not train_mask.any():
        raise ValueError("Training range is empty; adjust valid_start.")
    if not valid_mask.any():
        raise ValueError("Validation range is empty; adjust valid_start/test_start.")
    if not test_mask.any():
        raise ValueError("Test range is empty; adjust test_start.")

    return train_mask, valid_mask, test_mask
