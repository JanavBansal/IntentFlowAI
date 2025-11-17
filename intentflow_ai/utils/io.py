"""IO helpers for interacting with the local parquet lake."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq

from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.data.universe import apply_membership_filter, load_universe
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


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


def load_price_parquet(
    path: Path | None = None,
    *,
    allow_fallback: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    cfg: Settings | None = None,
) -> pd.DataFrame:
    """Load the canonical price parquet into a normalized panel."""

    cfg = cfg or settings
    target = Path(path) if path else cfg.data_dir / "raw" / "price_confirmation" / "data.parquet"
    if not target.exists():
        raise FileNotFoundError(f"Price parquet not found at {target}")

    try:
        frame = pd.read_parquet(target)
    except Exception as exc:  # pragma: no cover - legacy parquet fallback
        logger.warning("pd.read_parquet failed; retrying with legacy pyarrow reader.", exc_info=exc)
        try:
            frame = pq.read_table(target, use_legacy_dataset=True).to_pandas()
        except Exception as arrow_exc:
            logger.warning("Legacy pyarrow reader failed; retrying pandas legacy mode.", exc_info=arrow_exc)
            try:
                frame = pd.read_parquet(target, use_legacy_dataset=True)
            except Exception as final_exc:
                if not allow_fallback:
                    raise RuntimeError(
                        "Unable to read price parquet and fallback disabled. Ensure full NIFTY200 dataset is present."
                    ) from final_exc
                logger.error("All parquet readers failed; attempting CSV fallback.", exc_info=final_exc)
                frame = _load_price_from_csv()

    frame.columns = [col.strip().lower() for col in frame.columns]
    required = {"date", "ticker", "close", "volume", "sector"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {', '.join(sorted(missing))}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].astype(str).str.strip()

    start_dt = pd.to_datetime(start_date or getattr(cfg, "price_start", None)).tz_localize(None) if (
        start_date or getattr(cfg, "price_start", None)
    ) else None
    end_dt = pd.to_datetime(end_date or getattr(cfg, "price_end", None)).tz_localize(None) if (
        end_date or getattr(cfg, "price_end", None)
    ) else None
    if start_dt is not None:
        frame = frame.loc[frame["date"] >= start_dt]
    if end_dt is not None:
        frame = frame.loc[frame["date"] <= end_dt]

    universe = None
    try:
        universe_path = _resolve_data_path(cfg.universe_file, cfg)
        membership_attr = getattr(cfg, "universe_membership_file", "")
        membership_path = _resolve_data_path(membership_attr, cfg) if membership_attr else None
        universe = load_universe(
            universe_path,
            start_date=start_dt,
            end_date=end_dt,
            membership_path=membership_path if membership_path and membership_path.exists() else None,
        )
        allowed = set(universe["ticker_nse"].astype(str).str.upper())
        before = len(frame)
        filtered = frame[frame["ticker"].str.upper().isin(allowed)]
        if filtered.empty and before > 0:
            logger.warning("Universe filter removed all rows; retaining original frame for fallback data.")
        else:
            frame = filtered
            if before != len(frame):
                logger.info("Applied universe filter", extra={"dropped": before - len(frame), "kept": len(frame)})
    except Exception as exc:  # pragma: no cover - optional during tests
        logger.warning("Unable to apply static universe filter", exc_info=exc)

    membership_cols = {"ticker", "start_date", "end_date"}
    if universe is not None and membership_cols.issubset(universe.columns):
        try:
            before = len(frame)
            filtered = apply_membership_filter(frame, universe[list(membership_cols)])
            if filtered.empty and before > 0:
                logger.warning("Membership filter removed all rows; retaining original frame.")
            else:
                frame = filtered
                if before != len(frame):
                    logger.info(
                        "Applied membership history filter",
                        extra={"dropped": before - len(frame), "kept": len(frame)},
                    )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to apply membership history filter", exc_info=exc)

    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_price_from_csv() -> pd.DataFrame:
    """Fallback loader when parquet snapshots are unavailable."""

    csv_dir = settings.data_dir / "raw" / "prices"
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Parquet load failed and no CSV backups were found under {csv_dir}."
        )

    frames = []
    for path in files:
        df = pd.read_csv(path, parse_dates=["date"])
        if "ticker" not in df.columns:
            df["ticker"] = path.stem.upper()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    return combined


def _resolve_data_path(relative: str, cfg: Settings | None = None) -> Path:
    cfg = cfg or settings
    rel = Path(relative)
    if rel.is_absolute():
        return rel
    return cfg.data_dir / rel
