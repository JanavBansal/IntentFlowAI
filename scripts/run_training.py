"""Train LightGBM on parquet-backed data and persist experiment artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config import Settings
from intentflow_ai.pipelines import TrainingPipeline
from intentflow_ai.utils import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IntentFlow AI training pipeline.")
    parser.add_argument("--live", action="store_true", help="Trigger live ingestion before training.")
    parser.add_argument(
        "--no-regime-filter",
        action="store_true",
        help="Disable regime-specific training and metrics.",
    )
    parser.add_argument(
        "--experiment",
        default="v0",
        help="Experiment subdirectory name under experiments/ (default: v0).",
    )
    parser.add_argument("--max-tickers", type=int, help="Randomly sample this many tickers before training.")
    parser.add_argument("--valid-start", help="Override validation split start date (YYYY-MM-DD).")
    parser.add_argument("--test-start", help="Override test split start date (YYYY-MM-DD).")
    return parser.parse_args()


def fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.3f}"


def print_metrics_table(metrics: dict) -> None:
    columns = ["roc_auc", "pr_auc", "precision_at_10", "precision_at_20", "hit_rate_0.6"]
    header = "split".ljust(10) + " ".join(col.rjust(16) for col in columns)
    print("\nSplit metrics")
    print(header)
    for split in ["train", "valid", "test", "overall"]:
        if split not in metrics:
            continue
        row = split.ljust(10)
        for col in columns:
            row += fmt(metrics[split].get(col, float("nan"))).rjust(16)
        print(row)

    by_regime = metrics.get("by_regime")
    if by_regime:
        for regime, split_metrics in by_regime.items():
            print(f"\nRegime: {regime}")
            print(header)
            for split, values in split_metrics.items():
                row = split.ljust(10)
                for col in columns:
                    row += fmt(values.get(col, float("nan"))).rjust(16)
                print(row)


def main() -> None:
    args = parse_args()
    cfg = Settings()
    if args.valid_start:
        cfg = replace(cfg, valid_start=args.valid_start)
    if args.test_start:
        cfg = replace(cfg, test_start=args.test_start)

    pipeline = TrainingPipeline(cfg=cfg, regime_filter=not args.no_regime_filter, use_live_sources=args.live)

    tickers_subset = None
    if args.max_tickers:
        prices = load_price_parquet()
        unique = np.array(sorted(prices["ticker"].unique()))
        if unique.size == 0:
            raise ValueError("No tickers available in price parquet.")
        rng = np.random.default_rng(cfg.lgbm_seed)
        if args.max_tickers < unique.size:
            tickers_subset = rng.choice(unique, size=args.max_tickers, replace=False)
        else:
            tickers_subset = unique
        print(f"Using {len(tickers_subset)} tickers out of {unique.size}")

    result = pipeline.run(live=args.live, tickers_subset=tickers_subset)

    exp_dir = Path("experiments") / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "models": result["models"],
        "regime_classifier": result.get("regime_classifier"),
        "feature_columns": result["feature_columns"],
    }
    model_path = exp_dir / "lgb.pkl"
    dump(model_bundle, model_path)

    metrics_path = exp_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    preds = result["training_frame"][["date", "ticker", "label", "excess_fwd"]].copy()
    preds["proba"] = result["probabilities"].values
    preds = preds[["date", "ticker", "proba", "label", "excess_fwd"]]
    preds_path = exp_dir / "preds.csv"
    preds.to_csv(preds_path, index=False)

    logger.info(
        "Experiment artifacts written",
        extra={
            "metrics": result["metrics"],
            "model_path": str(model_path),
            "preds_path": str(preds_path),
        },
    )

    print_metrics_table(result["metrics"])
    if result.get("tickers_used"):
        print(f"\nTickers used: {len(result['tickers_used'])}")
    print(f"\nArtifacts saved to {exp_dir} (model={model_path}, metrics={metrics_path})")


if __name__ == "__main__":
    main()
