"""CLI helper for running the training pipeline end-to-end."""

from __future__ import annotations

import pandas as pd

from intentflow_ai.config import Settings
from intentflow_ai.pipelines import TrainingPipeline
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def load_placeholder_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return a minimal synthetic dataset so the pipeline can run."""

    idx = pd.date_range("2023-01-01", periods=250, freq="B")
    data = pd.DataFrame(
        {
            "close": 100 + pd.Series(range(len(idx)), index=idx).rolling(5).mean().fillna(0),
            "volume": 1_000_000,
        },
        index=idx,
    )
    target = (pd.Series(range(len(idx)), index=idx) % 2).astype(int)
    return data, target


def main() -> None:
    settings = Settings()
    dataset, target = load_placeholder_dataset()
    pipeline = TrainingPipeline(cfg=settings)
    artifact = pipeline.run(dataset, target)
    logger.info("Artifacts ready", extra={"keys": list(artifact.keys())})


if __name__ == "__main__":
    main()
