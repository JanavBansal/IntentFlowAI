"""Run walk-forward validation with anchored origin and expanding windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.features import FeatureEngineer
from intentflow_ai.modeling import LightGBMTrainer, ModelEvaluator
from intentflow_ai.validation import (
    WalkForwardConfig,
    compute_walk_forward_stability,
    evaluate_walk_forward_fold,
    generate_walk_forward_folds,
    summarize_walk_forward_results,
)
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name under experiments/ (e.g., v_universe_sanity)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML config (optional)",
    )
    parser.add_argument(
        "--train-start",
        default="2018-06-01",
        help="Anchored origin date for training (default: 2018-06-01)",
    )
    parser.add_argument(
        "--valid-duration",
        type=int,
        default=180,
        help="Validation period duration in days (default: 180 = 6 months)",
    )
    parser.add_argument(
        "--test-duration",
        type=int,
        default=180,
        help="Test period duration in days (default: 180 = 6 months)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=90,
        help="Step size between folds in days (default: 90 = 3 months)",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=10,
        help="Embargo period between splits in days (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load experiment config if provided
    exp_cfg = None
    if args.config:
        exp_cfg = load_experiment_config(args.config)
        cfg = apply_experiment_overrides(Settings(), exp_cfg)
        import intentflow_ai.config.settings as settings_module

        settings_module.settings = cfg
    else:
        cfg = settings

    experiment_name = args.experiment
    exp_dir = cfg.experiments_dir / experiment_name

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    logger.info(f"Running walk-forward validation for experiment: {experiment_name}")

    # Load training data
    train_path = exp_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run training first.")

    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded training data: {len(train_df)} rows")

    # Build features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.build(train_df)
    logger.info(f"Built {len(features.columns)} features")

    # Get labels and dates
    if "label" not in train_df.columns:
        raise ValueError("Training data must have 'label' column")
    labels = train_df["label"]
    dates = pd.to_datetime(train_df["date"])

    # Get excess returns for IC calculation
    excess_returns = train_df.get("excess_fwd", None)

    # Create walk-forward config
    wf_cfg = WalkForwardConfig(
        train_start=args.train_start,
        valid_duration_days=args.valid_duration,
        test_duration_days=args.test_duration,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        horizon_days=cfg.signal_horizon_days,
    )

    # Generate folds
    logger.info("Generating walk-forward folds...")
    folds = generate_walk_forward_folds(train_df, wf_cfg, date_col="date")

    if len(folds) == 0:
        raise ValueError("No valid folds generated. Check configuration.")

    # Create trainer and evaluator
    trainer = LightGBMTrainer(cfg.lgbm)
    evaluator = ModelEvaluator(horizon_days=cfg.signal_horizon_days)

    # Evaluate each fold
    logger.info(f"Evaluating {len(folds)} folds...")
    fold_results = []

    for fold in folds:
        logger.info(f"Evaluating fold {fold.fold_idx}...")

        # Get excess returns for this fold's test set
        test_excess = None
        if excess_returns is not None:
            test_excess = excess_returns.loc[fold.test_mask]

        # Evaluate fold
        fold_result = evaluate_walk_forward_fold(
            fold,
            features,
            labels,
            trainer,
            evaluator,
            excess_returns=excess_returns,
        )

        fold_results.append(fold_result)

    # Create summary
    summary_df = summarize_walk_forward_results(fold_results)
    summary_path = exp_dir / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved walk-forward summary to {summary_path}")

    # Compute stability metrics
    stability_metrics = {}
    for metric in ["roc_auc", "ic", "rank_ic"]:
        for split in ["valid", "test"]:
            stability = compute_walk_forward_stability(fold_results, metric_name=metric, split_name=split)
            if "error" not in stability:
                stability_metrics[f"{split}_{metric}"] = stability

    stability_path = exp_dir / "walk_forward_stability.json"
    stability_path.write_text(json.dumps(stability_metrics, indent=2))
    logger.info(f"Saved stability metrics to {stability_path}")

    # Print results
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)

    print("\nSummary by Fold:")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("STABILITY ANALYSIS")
    print("=" * 80)

    for key, stability in stability_metrics.items():
        if "error" in stability:
            continue
        print(f"\n{key.upper()}:")
        print(f"  Mean: {stability['mean']:.4f}")
        print(f"  Std:  {stability['std']:.4f}")
        print(f"  Min:  {stability['min']:.4f}")
        print(f"  Max:  {stability['max']:.4f}")
        print(f"  CV:   {stability['coefficient_of_variation']:.4f}")
        print(f"  Stability Score: {stability['stability_score']:.4f}")
        print(f"  Is Stable: {'✅' if stability['is_stable'] else '❌'}")

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    test_roc_stability = stability_metrics.get("test_roc_auc", {})
    if test_roc_stability and "mean" in test_roc_stability:
        mean_test_roc = test_roc_stability["mean"]
        std_test_roc = test_roc_stability["std"]
        cv_test_roc = test_roc_stability.get("coefficient_of_variation", 0)

        print(f"\nTest ROC AUC: {mean_test_roc:.4f} ± {std_test_roc:.4f}")
        print(f"Coefficient of Variation: {cv_test_roc:.4f}")

        if mean_test_roc > 0.55 and cv_test_roc < 0.20:
            print("✅ Model shows consistent performance across folds")
        elif mean_test_roc > 0.50:
            print("⚠️  Model performance is above random but may be unstable")
        else:
            print("❌ Model performance is near random or unstable")

    print("\n" + "=" * 80)
    print("Walk-forward validation complete!")
    print("=" * 80)
    print(f"\nResults saved to: {exp_dir}")


if __name__ == "__main__":
    main()

