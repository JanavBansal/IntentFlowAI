"""IO helpers for interacting with the local parquet lake."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from intentflow_ai.config.settings import settings


def write_parquet_partition(target_dir: Path, records: Iterable[dict]) -> None:
    """Persist iterable of dict rows into a partition directory.

    In early scaffolding we simply convert to a DataFrame. Later this helper
    can enforce schemas, add metadata, and upload to cloud object storage.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame([{"placeholder": None}])  # ensures files exist downstream
    target_path = target_dir / "data.parquet"
    df.to_parquet(target_path, index=False)


def read_parquet_dataset(paths: Sequence[Path]) -> pd.DataFrame:
    """Load multiple parquet files and concatenate them."""

    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def load_price_parquet(path: Path | None = None) -> pd.DataFrame:
    """Load the canonical price parquet into a normalized panel."""

    target = Path(path) if path else settings.data_dir / "raw" / "price_confirmation" / "data.parquet"
    if not target.exists():
        raise FileNotFoundError(f"Price parquet not found at {target}")

    frame = pd.read_parquet(target)
    frame.columns = [col.strip().lower() for col in frame.columns]
    required = {"date", "ticker", "close", "volume", "sector"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {', '.join(sorted(missing))}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)
