"""IO helpers for interacting with the local parquet lake."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


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
