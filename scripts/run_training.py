"""Train LightGBM on parquet-backed data and persist experiment artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config import Settings
from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.meta_labeling import MetaLabelConfig
from intentflow_ai.modeling.visuals import save_classification_plots
from intentflow_ai.pipelines import TrainingPipeline
from intentflow_ai.sanity import DataScopeChecks
from intentflow_ai.utils.io import load_price_parquet
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
        default=None,
        help="Experiment subdirectory name under experiments/ (default: config id or v0).",
    )
    parser.add_argument(
        "--config",
        default="config/experiments/v_universe_sanity.yaml",
        help="Experiment YAML describing splits/hyperparameters.",
    )
    parser.add_argument("--max-tickers", type=int, help="Randomly sample this many tickers before training.")
    parser.add_argument("--valid-start", help="Override validation split start date (YYYY-MM-DD).")
    parser.add_argument("--test-start", help="Override test split start date (YYYY-MM-DD).")
    parser.add_argument(
        "--leak-test",
        action="store_true",
        help="Shuffle labels before training to confirm metrics collapse.",
    )
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
    exp_cfg = load_experiment_config(args.config) if args.config else None
    cfg = apply_experiment_overrides(Settings(), exp_cfg)
    if args.valid_start:
        cfg = replace(cfg, valid_start=args.valid_start)
    if args.test_start:
        cfg = replace(cfg, test_start=args.test_start)
    # Propagate overrides to global settings users (feature lookups, IO helpers).
    import intentflow_ai.config.settings as settings_module
    from intentflow_ai.features.engineering import _sector_lookup

    settings_module.settings = cfg
    _sector_lookup.cache_clear()

    meta_cfg: MetaLabelConfig | None = None
    if exp_cfg:
        meta_block = exp_cfg.get("meta_labeling", {})
        if meta_block:
            meta_cfg = MetaLabelConfig(
                enabled=bool(meta_block.get("enabled", False)),
                horizon_days=meta_block.get("horizon_days", cfg.signal_horizon_days),
                success_threshold=meta_block.get("success_threshold", cfg.target_excess_return),
                min_signal_proba=meta_block.get("min_signal_proba", 0.0),
                random_state=meta_block.get("random_state", cfg.lgbm_seed),
                proba_col=meta_block.get("proba_col", "proba"),
                output_col=meta_block.get("output_col", "meta_proba"),
            )
    if meta_cfg is None:
        meta_cfg = MetaLabelConfig(enabled=False, horizon_days=cfg.signal_horizon_days, random_state=cfg.lgbm_seed)

    experiment_name = args.experiment or (exp_cfg.id if exp_cfg else "v0")

    pipeline = TrainingPipeline(
        cfg=cfg,
        regime_filter=not args.no_regime_filter,
        use_live_sources=args.live,
        leak_test=args.leak_test,
        meta_labeling_cfg=meta_cfg,
    )

    tickers_subset = None
    if args.max_tickers:
        prices = load_price_parquet(
            allow_fallback=False,
            start_date=cfg.price_start,
            end_date=cfg.price_end,
        )
        unique = np.array(sorted(prices["ticker"].unique()))
        if unique.size == 0:
            raise ValueError("No tickers available in price parquet.")
        rng = np.random.default_rng(cfg.lgbm_seed)
        if args.max_tickers < unique.size:
            tickers_subset = rng.choice(unique, size=args.max_tickers, replace=False)
        else:
            tickers_subset = unique
        print(f"Using {len(tickers_subset)} tickers out of {unique.size}")
    if args.leak_test:
        print("Leak test enabled: labels will be shuffled before training.")

    result = pipeline.run(live=args.live, tickers_subset=tickers_subset)

    exp_dir = Path("experiments") / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = save_classification_plots(
        result["training_frame"]["label"],
        result["probabilities"],
        result["metrics"]["overall"].get("hit_curve", []),
        exp_dir / "plots",
    )
    if plot_paths:
        relative = {name: str(path.relative_to(exp_dir)) for name, path in plot_paths.items()}
        result["metrics"]["plots"] = relative

    model_bundle = {
        "models": result["models"],
        "regime_classifier": result.get("regime_classifier"),
        "feature_columns": result["feature_columns"],
    }
    meta_payload = result.get("meta", {})
    if meta_payload.get("model") is not None:
        model_bundle["meta_model"] = meta_payload["model"]
        model_bundle["meta_feature_columns"] = meta_payload.get("feature_columns")
        model_bundle["meta_config"] = result.get("meta_config")
    model_path = exp_dir / "lgb.pkl"
    dump(model_bundle, model_path)

    train_frame = result["training_frame"]
    train_frame_path = exp_dir / "train.parquet"
    train_frame.to_parquet(train_frame_path, index=False)

    preds_cols = ["date", "ticker", "label", "excess_fwd", "proba"]
    if "meta_proba" in train_frame.columns:
        preds_cols.append("meta_proba")
    preds = train_frame[preds_cols].copy()
    preds["proba"] = result["probabilities"].values
    preds_path = exp_dir / "preds.csv"
    preds.to_csv(preds_path, index=False)

    train_mask = pd.Series(result["splits"]["train"], index=train_frame.index, dtype=bool)
    data_scope = DataScopeChecks(min_train_tickers=cfg.min_train_tickers)
    snapshot_path = exp_dir / "universe_snapshot.csv"
    data_scope.validate_and_snapshot(train_frame, train_mask, output_path=snapshot_path)

    fi = result.get("feature_importances", {})
    if fi:
        fi_path = exp_dir / "feature_importance.csv"
        (
            pd.Series(fi, name="importance")
            .sort_values(ascending=False)
            .to_csv(fi_path, header=True)
        )

    shap_plot_path = exp_dir / "plots" / "shap_summary.png"
    try:  # Optional dependency
        import shap
        import matplotlib.pyplot as plt

        sample = train_frame[result["feature_columns"]].sample(
            n=min(500, len(train_frame)),
            random_state=cfg.lgbm_seed,
        )
        explainer = shap.TreeExplainer(result["model"])
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values, sample, show=False, plot_type="violin")
        plt.tight_layout()
        shap_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        result["metrics"].setdefault("plots", {})["shap_summary"] = str(shap_plot_path.relative_to(exp_dir))
    except ImportError:
        logger.warning("SHAP not installed; skipping shap summary plot.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to compute SHAP summary", exc_info=exc)

    metrics_path = exp_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    params_path = exp_dir / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(
            {
                "settings": asdict(cfg),
                "meta_labeling": asdict(meta_cfg),
                "experiment_config": exp_cfg.data if exp_cfg else {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

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
