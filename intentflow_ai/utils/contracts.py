"""Lightweight schema validation helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def validate_schema(df: pd.DataFrame, required: Dict[str, str]) -> pd.DataFrame:
    """Coerce dataframe to expected dtypes and ensure columns exist."""

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    coerced = df.copy()
    for col, dtype in required.items():
        if dtype.startswith("datetime"):
            coerced[col] = pd.to_datetime(coerced[col], errors="coerce")
        elif dtype in ("float", "float64"):
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce").astype(float)
        elif dtype in ("string", "str"):
            coerced[col] = coerced[col].astype("string")
        else:
            coerced[col] = coerced[col].astype(dtype, errors="raise")
        if coerced[col].isnull().any():
            raise ValueError(f"Column {col} contains nulls after coercion.")
    return coerced
