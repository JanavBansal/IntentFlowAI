"""Run ingestion, training, and scoring in one go to produce dashboard-ready outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings
from intentflow_ai.data.ingestion import DataIngestionWorkflow
from intentflow_ai.features import FeatureEngineer
from intentflow_ai.pipelines import ScoringPipeline, TrainingPipeline
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def run_live_ingestion() -> Path:
    workflow = DataIngestionWorkflow()
    workflow.run()
    price_path = workflow.output_dir / "raw" / "price_confirmation" / "data.parquet"
    logger.info("Ingestion complete", extra={"price_parquet": str(price_path)})
    return price_path


def run_training_step(exp_dir: Path) -> dict:
    pipeline = TrainingPipeline(use_live_sources=False)
    result = pipeline.run(live=False)

    exp_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "models": result["models"],
        "regime_classifier": result.get("regime_classifier"),
        "feature_columns": result["feature_columns"],
    }
    model_path = exp_dir / "lgb.pkl"
    dump(model_bundle, model_path)

    metrics_path = exp_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(result["metrics"], fh, indent=2)

    preds_path = exp_dir / "preds.csv"
    preds = result["training_frame"][["date", "ticker", "label", "excess_fwd"]].copy()
    preds["proba"] = result["probabilities"].values
    preds.to_csv(preds_path, index=False)

    logger.info(
        "Training artifacts saved",
        extra={"metrics": result["metrics"], "model_path": str(model_path), "preds_path": str(preds_path)},
    )
    return result


def run_scoring_step(result: dict, exp_dir: Path) -> Path:
    panel = load_price_parquet()
    latest_date = panel["date"].max()
    window = panel[panel["date"] >= latest_date - pd.Timedelta(days=60)].reset_index(drop=True)
    if window.empty:
        raise ValueError("No observations available within the scoring window.")

    scoring = ScoringPipeline(
        feature_engineer=FeatureEngineer(),
        models=result["models"],
        feature_columns=result["feature_columns"],
        regime_classifier=result.get("regime_classifier"),
    )
    signals = scoring.run(window)
    signals = signals.merge(window[["date", "ticker", "sector"]], on=["date", "ticker"], how="left")
    signals = signals[["date", "ticker", "sector", "proba", "rank"]]

    top_path = exp_dir / "top_signals.csv"
    signals.to_csv(top_path, index=False)
    logger.info("Scoring complete", extra={"top_signals": str(top_path), "rows": len(signals)})
    return top_path


def main() -> None:
    exp_dir = settings.experiments_dir / "v0"
    run_live_ingestion()
    training_result = run_training_step(exp_dir)
    top_path = run_scoring_step(training_result, exp_dir)

    print(
        "Pipeline finished.\n"
        f"- Metrics: { (exp_dir / 'metrics.json').resolve() }\n"
        f"- Model:   { (exp_dir / 'lgb.pkl').resolve() }\n"
        f"- Signals: { top_path.resolve() }"
    )


if __name__ == "__main__":
    main()
