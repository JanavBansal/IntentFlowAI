"""Generate ranked swing signals using the latest parquet snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.features import FeatureEngineer
from intentflow_ai.pipelines import ScoringPipeline
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    exp_dir = Path("experiments/v0")
    model_path = exp_dir / "lgb.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
    bundle = load(model_path)
    models = bundle.get("models", {})
    regime_classifier = bundle.get("regime_classifier")
    feature_columns = bundle.get("feature_columns")

    panel = load_price_parquet()
    latest_date = panel["date"].max()
    window_start = latest_date - pd.Timedelta(days=60)
    window = panel[panel["date"] >= window_start].reset_index(drop=True)
    if window.empty:
        raise ValueError("No records available in the requested scoring window.")

    scoring = ScoringPipeline(
        feature_engineer=FeatureEngineer(),
        models=models,
        feature_columns=feature_columns,
        regime_classifier=regime_classifier,
    )
    signals = scoring.run(window)
    signals = signals.merge(
        window[["date", "ticker", "sector"]],
        on=["date", "ticker"],
        how="left",
    )
    signals = signals[["date", "ticker", "sector", "proba", "rank"]]

    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "top_signals.csv"
    signals.to_csv(output_path, index=False)

    logger.info("Top signals written", extra={"path": str(output_path), "rows": len(signals)})
    print(signals.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
