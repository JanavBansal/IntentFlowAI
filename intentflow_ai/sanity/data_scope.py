"""Ticker coverage and universe snapshot helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataScopeChecks:
    """Validate ticker coverage and persist experiment snapshots."""

    min_train_tickers: int = 180

    def validate_and_snapshot(
        self,
        frame: pd.DataFrame,
        train_mask: pd.Series,
        *,
        date_col: str = "date",
        ticker_col: str = "ticker",
        output_path: Path,
    ) -> int:
        """Assert that training coverage meets thresholds and write snapshot CSV."""

        if frame.empty:
            raise ValueError("Training frame is empty; cannot perform data scope checks.")

        unique_train = frame.loc[train_mask, ticker_col].nunique()
        logger.info("Data scope check", extra={"unique_train_tickers": unique_train})
        if unique_train < self.min_train_tickers:
            raise AssertionError(
                f"Training tickers below threshold: {unique_train} < {self.min_train_tickers}. "
                "Verify full NIFTY200 parquet ingestion."
            )

        snapshot = (
            frame[[date_col, ticker_col]]
            .drop_duplicates()
            .assign(in_universe_flag=True)
            .sort_values([date_col, ticker_col])
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(output_path, index=False)
        logger.info("Universe snapshot written", extra={"rows": len(snapshot), "path": str(output_path)})
        return unique_train
