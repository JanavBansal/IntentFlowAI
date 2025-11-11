"""CLI helper to run the ingestion workflow and materialize parquet outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings
from intentflow_ai.data.coverage import price_coverage_report
from intentflow_ai.data.ingestion import DataIngestionWorkflow
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data ingestion jobs.")
    parser.add_argument("--experiment", default="latest", help="Experiment folder for coverage outputs.")
    args = parser.parse_args()

    workflow = DataIngestionWorkflow()
    workflow.run()
    price_path = workflow.output_dir / "raw" / "price_confirmation" / "data.parquet"
    logger.info("Price parquet ready", extra={"path": str(price_path)})
    print(f"Parquet written to: {price_path.relative_to(workflow.cfg.project_root)}")

    prices = pd.read_parquet(price_path)
    coverage = price_coverage_report(prices, min_days=settings.min_trading_days)
    exp_dir = settings.experiments_dir / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    coverage["per_ticker"].to_csv(exp_dir / "coverage_prices.csv", index=False)
    (exp_dir / "coverage_prices.json").write_text(json.dumps(coverage["summary"], indent=2), encoding="utf-8")
    logger.info("Coverage report written", extra={"experiment": args.experiment})


if __name__ == "__main__":
    main()
