"""Walk-forward validation with anchored origin and expanding windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    train_start: str  # Anchored origin date (e.g., "2018-06-01")
    train_end: Optional[str] = None  # Optional fixed training end date
    valid_duration_days: int = 180  # Validation period duration (6 months)
    test_duration_days: int = 180  # Test period duration (6 months)
    step_days: int = 90  # Step size between folds (3 months)
    embargo_days: int = 10  # Embargo period between train/valid and valid/test
    horizon_days: int = 10  # Prediction horizon (for additional embargo)
    min_train_days: int = 252  # Minimum training period (1 year)
    min_valid_days: int = 60  # Minimum validation period
    min_test_days: int = 60  # Minimum test period


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation."""

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: pd.Series
    valid_mask: pd.Series
    test_mask: pd.Series


def generate_walk_forward_folds(
    df: pd.DataFrame,
    cfg: WalkForwardConfig,
    *,
    date_col: str = "date",
) -> List[WalkForwardFold]:
    """Generate walk-forward validation folds with anchored origin.

    The training window always starts from train_start (anchored origin) and
    expands with each fold. Validation and test windows move forward in time.

    Example:
        Fold 1: Train[2018-06:2020-06] | Valid[2020-07:2020-12] | Test[2021-01:2021-06]
        Fold 2: Train[2018-06:2020-12] | Valid[2021-01:2021-06] | Test[2021-07:2021-12]
        Fold 3: Train[2018-06:2021-06] | Valid[2021-07:2021-12] | Test[2022-01:2022-06]

    Args:
        df: DataFrame with date column
        cfg: Walk-forward configuration
        date_col: Column name for dates

    Returns:
        List of WalkForwardFold objects
    """
    if date_col not in df.columns:
        raise ValueError(f"DataFrame must have '{date_col}' column")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    unique_dates = dates.dropna().sort_values().unique()

    if len(unique_dates) == 0:
        raise ValueError("No valid dates found in DataFrame")

    train_start_dt = pd.to_datetime(cfg.train_start)
    train_end_dt = pd.to_datetime(cfg.train_end) if cfg.train_end else None

    # Find first available date >= train_start
    available_dates = unique_dates[unique_dates >= train_start_dt]
    if len(available_dates) == 0:
        raise ValueError(f"No dates available after train_start {cfg.train_start}")

    # Start with first fold - set initial training end to be after minimum training period
    # We need at least min_train_days of training data before we can start validation
    if train_end_dt is None:
        # Find the date that gives us at least min_train_days of unique trading days
        # Count unique dates from train_start and find where we have enough
        dates_from_start = available_dates[available_dates >= train_start_dt]
        if len(dates_from_start) < cfg.min_train_days:
            raise ValueError(
                f"Not enough data for minimum training period. "
                f"Need {cfg.min_train_days} unique trading days from {cfg.train_start}, "
                f"but only have {len(dates_from_start)} days available (data goes to {available_dates[-1]})"
            )
        # Set current_train_end to the date that gives us exactly min_train_days of training data
        # (or as close as possible)
        current_train_end = dates_from_start[cfg.min_train_days - 1]
    else:
        current_train_end = train_end_dt
    folds = []

    fold_idx = 1
    max_iterations = 100  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Calculate validation period
        valid_start = current_train_end + pd.Timedelta(days=cfg.embargo_days + cfg.horizon_days)
        valid_end = valid_start + pd.Timedelta(days=cfg.valid_duration_days)

        # Calculate test period
        test_start = valid_end + pd.Timedelta(days=cfg.embargo_days + cfg.horizon_days)
        test_end = test_start + pd.Timedelta(days=cfg.test_duration_days)

        # Check if we have enough data
        train_dates = unique_dates[(unique_dates >= train_start_dt) & (unique_dates <= current_train_end)]
        valid_dates = unique_dates[(unique_dates >= valid_start) & (unique_dates < valid_end)]
        test_dates = unique_dates[(unique_dates >= test_start) & (unique_dates < test_end)]

        train_days = len(train_dates)
        valid_days = len(valid_dates)
        test_days = len(test_dates)

        # Check minimum requirements
        if train_days < cfg.min_train_days:
            logger.warning(
                f"Fold {fold_idx}: Insufficient training data ({train_days} < {cfg.min_train_days} days). Stopping."
            )
            break

        # Allow smaller validation/test periods if we're at the end of data
        min_valid_required = max(cfg.min_valid_days // 2, 30)  # At least 30 days
        min_test_required = max(cfg.min_test_days // 2, 30)  # At least 30 days
        
        if valid_days < min_valid_required:
            logger.warning(
                f"Fold {fold_idx}: Insufficient validation data ({valid_days} < {min_valid_required} days). Skipping."
            )
            # Try to advance to next fold
            current_train_end = current_train_end + pd.Timedelta(days=cfg.step_days)
            continue

        if test_days < min_test_required:
            logger.warning(
                f"Fold {fold_idx}: Insufficient test data ({test_days} < {min_test_required} days). "
                f"Using available data ({test_days} days)."
            )
            # Continue with available test data if we have at least 30 days
            if test_days < 30:
                break

        # Create masks
        train_mask = (dates >= train_start_dt) & (dates <= current_train_end)
        valid_mask = (dates >= valid_start) & (dates < valid_end)
        test_mask = (dates >= test_start) & (dates < test_end)

        # Verify masks have data
        if not train_mask.any() or not valid_mask.any() or not test_mask.any():
            logger.warning(f"Fold {fold_idx}: Empty masks detected. Stopping.")
            break

        fold = WalkForwardFold(
            fold_idx=fold_idx,
            train_start=train_start_dt,
            train_end=current_train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            train_mask=train_mask,
            valid_mask=valid_mask,
            test_mask=test_mask,
        )

        folds.append(fold)
        logger.info(
            f"Fold {fold_idx}: Train[{train_start_dt.date()} to {current_train_end.date()}] "
            f"({train_days} days) | Valid[{valid_start.date()} to {valid_end.date()}] "
            f"({valid_days} days) | Test[{test_start.date()} to {test_end.date()}] ({test_days} days)"
        )

        # Advance to next fold
        current_train_end = current_train_end + pd.Timedelta(days=cfg.step_days)

        # Check if we've run out of data
        if current_train_end >= unique_dates[-1]:
            break

        fold_idx += 1

    if len(folds) == 0:
        raise ValueError("No valid folds generated. Check configuration and data availability.")

    logger.info(f"Generated {len(folds)} walk-forward folds")
    return folds


def evaluate_walk_forward_fold(
    fold: WalkForwardFold,
    features: pd.DataFrame,
    labels: pd.Series,
    trainer,
    evaluator,
    *,
    excess_returns: Optional[pd.Series] = None,
) -> Dict:
    """Evaluate a single walk-forward fold.

    Args:
        fold: WalkForwardFold object
        features: DataFrame with feature columns
        labels: Series with target labels
        trainer: Model trainer object with train() method
        evaluator: Model evaluator object with evaluate() method
        excess_returns: Optional Series with excess returns for IC calculation

    Returns:
        Dictionary with metrics for train, valid, and test sets
    """
    # Train model
    train_features = features.loc[fold.train_mask]
    train_labels = labels.loc[fold.train_mask]

    if train_features.empty or train_labels.empty:
        return {"error": "Empty training set"}

    model = trainer.train(train_features, train_labels)

    # Evaluate on validation set
    valid_features = features.loc[fold.valid_mask]
    valid_labels = labels.loc[fold.valid_mask]

    valid_metrics = {}
    if not valid_features.empty and not valid_labels.empty:
        valid_proba = pd.Series(
            model.predict_proba(valid_features)[:, 1],
            index=valid_labels.index,
        )
        valid_excess = excess_returns.loc[fold.valid_mask] if excess_returns is not None else None
        valid_metrics = evaluator.evaluate(
            valid_labels,
            valid_proba,
            excess_returns=valid_excess,
        )

    # Evaluate on test set
    test_features = features.loc[fold.test_mask]
    test_labels = labels.loc[fold.test_mask]

    test_metrics = {}
    if not test_features.empty and not test_labels.empty:
        test_proba = pd.Series(
            model.predict_proba(test_features)[:, 1],
            index=test_labels.index,
        )
        test_excess = excess_returns.loc[fold.test_mask] if excess_returns is not None else None
        test_metrics = evaluator.evaluate(
            test_labels,
            test_proba,
            excess_returns=test_excess,
        )

    return {
        "fold_idx": fold.fold_idx,
        "train_start": fold.train_start.isoformat(),
        "train_end": fold.train_end.isoformat(),
        "valid_start": fold.valid_start.isoformat(),
        "valid_end": fold.valid_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "train_samples": len(train_features),
        "valid_samples": len(valid_features),
        "test_samples": len(test_features),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }


def compute_walk_forward_stability(
    fold_results: List[Dict],
    metric_name: str = "roc_auc",
    split_name: str = "test",
) -> Dict:
    """Compute stability metrics across walk-forward folds.

    Args:
        fold_results: List of fold evaluation results
        metric_name: Metric to analyze (e.g., "roc_auc", "ic")
        split_name: Split to analyze ("train", "valid", or "test")

    Returns:
        Dictionary with stability statistics
    """
    metrics = []
    for fold_result in fold_results:
        split_metrics = fold_result.get(f"{split_name}_metrics", {})
        if metric_name in split_metrics:
            metrics.append(split_metrics[metric_name])

    if len(metrics) == 0:
        return {"error": f"No {metric_name} found in {split_name} metrics"}

    metrics_array = np.array(metrics)
    mean_metric = float(np.nanmean(metrics_array))
    std_metric = float(np.nanstd(metrics_array))
    min_metric = float(np.nanmin(metrics_array))
    max_metric = float(np.nanmax(metrics_array))

    # Coefficient of variation (std / mean)
    cv = std_metric / abs(mean_metric) if mean_metric != 0 else np.inf

    # Stability score: lower CV = more stable
    stability_score = 1.0 / (1.0 + cv) if not np.isinf(cv) else 0.0

    return {
        "metric": metric_name,
        "split": split_name,
        "n_folds": len(metrics),
        "mean": mean_metric,
        "std": std_metric,
        "min": min_metric,
        "max": max_metric,
        "coefficient_of_variation": cv,
        "stability_score": stability_score,
        "is_stable": cv < 0.20,  # Consider stable if CV < 20%
    }


def summarize_walk_forward_results(
    fold_results: List[Dict],
    key_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create summary DataFrame of walk-forward results.

    Args:
        fold_results: List of fold evaluation results
        key_metrics: List of metrics to include (default: ["roc_auc", "ic", "rank_ic"])

    Returns:
        DataFrame with summary statistics
    """
    if key_metrics is None:
        key_metrics = ["roc_auc", "ic", "rank_ic"]

    rows = []
    for fold_result in fold_results:
        row = {
            "fold_idx": fold_result.get("fold_idx"),
            "train_start": fold_result.get("train_start"),
            "train_end": fold_result.get("train_end"),
            "valid_start": fold_result.get("valid_start"),
            "valid_end": fold_result.get("valid_end"),
            "test_start": fold_result.get("test_start"),
            "test_end": fold_result.get("test_end"),
            "train_samples": fold_result.get("train_samples", 0),
            "valid_samples": fold_result.get("valid_samples", 0),
            "test_samples": fold_result.get("test_samples", 0),
        }

        # Add metrics for each split
        for split in ["train", "valid", "test"]:
            split_metrics = fold_result.get(f"{split}_metrics", {})
            for metric in key_metrics:
                row[f"{split}_{metric}"] = split_metrics.get(metric, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)

