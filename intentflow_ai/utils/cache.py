"""Simple parquet caching utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd


def cache_parquet(
    path: Path,
    key: str,
    producer: Callable[[], pd.DataFrame],
    ttl_hours: int = 12,
) -> pd.DataFrame:
    """Return cached parquet data if fresh; otherwise run producer and cache."""

    meta_path = path.with_suffix(".meta.json")
    now = time.time()
    ttl_seconds = ttl_hours * 3600
    if path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("key") == key and now - meta.get("timestamp", 0) <= ttl_seconds:
                return pd.read_parquet(path)
        except Exception:
            pass

    df = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta_path.write_text(json.dumps({"key": key, "timestamp": now}), encoding="utf-8")
    return df
