"""CLI helper to run the ingestion workflow and materialize parquet outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.data.ingestion import DataIngestionWorkflow
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    workflow = DataIngestionWorkflow()
    workflow.run()
    price_path = workflow.output_dir / "raw" / "price_confirmation" / "data.parquet"
    logger.info("Price parquet ready", extra={"path": str(price_path)})
    print(f"Parquet written to: {price_path.relative_to(workflow.cfg.project_root)}")


if __name__ == "__main__":
    main()
