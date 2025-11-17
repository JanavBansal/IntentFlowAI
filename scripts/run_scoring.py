"""Generate ranked swing signals using the latest parquet snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config import Settings
from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.features import FeatureEngineer
from intentflow_ai.pipelines import ScoringPipeline
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scoring pipeline for a given experiment.")
    parser.add_argument("--experiment", default=None, help="Experiment directory under experiments/")
    parser.add_argument(
        "--config",
        default="config/experiments/v_universe_sanity.yaml",
        help="Experiment YAML (for default experiment id).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.config) if args.config else None
    cfg = apply_experiment_overrides(Settings(), exp_cfg)
    import intentflow_ai.config.settings as settings_module
    from intentflow_ai.features.engineering import _sector_lookup

    settings_module.settings = cfg
    _sector_lookup.cache_clear()
    exp_dir = cfg.experiments_dir / (args.experiment or (exp_cfg.id if exp_cfg else "v0"))
    model_path = exp_dir / "lgb.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
    bundle = load(model_path)
    models = bundle.get("models", {})
    regime_classifier = bundle.get("regime_classifier")
    feature_columns = bundle.get("feature_columns")
    meta_model = bundle.get("meta_model")
    meta_feature_columns = bundle.get("meta_feature_columns")
    meta_config = bundle.get("meta_config")

    panel = load_price_parquet(
        allow_fallback=False,
        start_date=getattr(cfg, "price_start", None),
        end_date=getattr(cfg, "price_end", None),
        cfg=cfg,
    )
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
        meta_model=meta_model,
        meta_feature_columns=meta_feature_columns,
        meta_config=meta_config,
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
