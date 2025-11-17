"""Generate ranked swing signals using the latest parquet snapshot."""

from __future__ import annotations

import argparse
import json
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
from intentflow_ai.modeling import ExplanationConfig
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

    # Load training data for SHAP background
    background_data = None
    explanation_config = ExplanationConfig(enabled=True, max_features=10)
    
    # Try train.parquet first, then fallback to preds.csv
    train_frame_path = exp_dir / "train.parquet"
    preds_path = exp_dir / "preds.csv"
    
    if train_frame_path.exists():
        try:
            train_frame = pd.read_parquet(train_frame_path)
            feature_engineer = FeatureEngineer()
            background_features = feature_engineer.build(train_frame).fillna(0.0)
            if feature_columns:
                background_features = background_features.reindex(columns=feature_columns, fill_value=0.0)
            background_data = background_features
            logger.info("Loaded training data for SHAP background from train.parquet", extra={"samples": len(background_data)})
        except Exception as exc:
            logger.warning("Failed to load train.parquet for SHAP, trying preds.csv", exc_info=exc)
            train_frame_path = None
    
    # Fallback to preds.csv if train.parquet not available
    if background_data is None and preds_path.exists():
        try:
            preds_frame = pd.read_csv(preds_path, parse_dates=["date"])
            # Use a sample of predictions as background
            sample_size = min(500, len(preds_frame))
            preds_sample = preds_frame.sample(n=sample_size, random_state=42) if len(preds_frame) > sample_size else preds_frame
            
            # Merge with price data to get full features
            preds_with_prices = preds_sample.merge(
                panel[["date", "ticker", "close", "open", "high", "low", "volume", "sector"]],
                on=["date", "ticker"],
                how="left"
            )
            
            feature_engineer = FeatureEngineer()
            background_features = feature_engineer.build(preds_with_prices).fillna(0.0)
            if feature_columns:
                background_features = background_features.reindex(columns=feature_columns, fill_value=0.0)
            background_data = background_features
            logger.info("Loaded training data for SHAP background from preds.csv", extra={"samples": len(background_data)})
        except Exception as exc:
            logger.warning("Failed to load preds.csv for SHAP, explanations disabled", exc_info=exc)
            explanation_config.enabled = False
    
    if background_data is None:
        logger.warning("No background data available for SHAP, explanations disabled")
        explanation_config.enabled = False

    scoring = ScoringPipeline(
        feature_engineer=FeatureEngineer(),
        models=models,
        feature_columns=feature_columns,
        regime_classifier=regime_classifier,
        meta_model=meta_model,
        meta_feature_columns=meta_feature_columns,
        meta_config=meta_config,
        explanation_config=explanation_config,
        background_data=background_data,
    )
    signals = scoring.run(window)
    
    # Ensure sector is present (should already be from scoring pipeline)
    if "sector" not in signals.columns:
        signals = signals.merge(
            window[["date", "ticker", "sector"]],
            on=["date", "ticker"],
            how="left",
        )
    
    # Select output columns (include explanation columns if present)
    output_cols = ["date", "ticker", "sector", "proba", "rank"]
    if "top_features" in signals.columns:
        output_cols.append("top_features")
    if "rationale" in signals.columns:
        output_cols.append("rationale")
    if "shap_values" in signals.columns:
        output_cols.append("shap_values")
    
    # Only include columns that exist
    available_cols = [col for col in output_cols if col in signals.columns]
    signals = signals[available_cols]
    
    # Log what was generated
    if "top_features" in signals.columns:
        logger.info("SHAP explanations included in output")
    else:
        logger.info("SHAP explanations not available (background data may be missing)")

    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "top_signals.csv"
    
    # Convert complex types to JSON strings for CSV compatibility
    signals_output = signals.copy()
    for col in ["top_features", "shap_values"]:
        if col in signals_output.columns:
            signals_output[col] = signals_output[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    
    signals_output.to_csv(output_path, index=False)

    logger.info("Top signals written", extra={"path": str(output_path), "rows": len(signals)})
    
    # Display summary
    display_cols = ["date", "ticker", "sector", "proba", "rank"]
    if "rationale" in signals.columns:
        display_cols.append("rationale")
    display_df = signals[display_cols].head(10) if all(c in signals.columns for c in display_cols) else signals.head(10)
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
