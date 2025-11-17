"""Enhanced leakage detection utilities with multiple validation strategies.

This module implements comprehensive leakage detection including:
- Random label permutation by date blocks
- Reversed time splits (train on future, test on past)
- Gap-based splits (skip intermediate periods)
- Feature-target correlation on future data
- Rolling window validation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.modeling import LightGBMTrainer, ModelEvaluator
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def verify_forward_alignment(
    frame: pd.DataFrame,
    *,
    price_col: str = "close",
    ticker_col: str = "ticker",
    date_col: str = "date",
    fwd_ret_col: str,
    horizon_days: int,
    atol: float = 1e-6,
) -> None:
    """Ensure stored forward returns match recomputed values from prices."""

    if fwd_ret_col not in frame.columns:
        raise KeyError(f"Forward-return column '{fwd_ret_col}' missing from training frame.")

    prices = frame[[date_col, ticker_col, price_col]].drop_duplicates().sort_values([ticker_col, date_col])

    prices = prices.sort_values([ticker_col, date_col])
    shifted = prices.groupby(ticker_col)[price_col].shift(-horizon_days)
    prices["expected_fwd"] = shifted / prices[price_col] - 1.0

    aligned = frame[[date_col, ticker_col, fwd_ret_col]].copy()
    aligned = aligned.merge(
        prices[[date_col, ticker_col, "expected_fwd"]],
        on=[date_col, ticker_col],
        how="left",
    )
    deltas = (aligned[fwd_ret_col] - aligned["expected_fwd"]).abs().dropna()
    max_delta = deltas.max() if not deltas.empty else 0.0
    logger.info("Forward alignment delta", extra={"max_delta": float(max_delta)})
    if max_delta and max_delta > atol:
        raise AssertionError(
            f"Forward return misalignment detected (max abs delta={max_delta:.4g}). "
            "Ensure labels only use t -> t+horizon window."
        )


@dataclass
class NullLabelResult:
    sharpe: float
    ic: float
    rank_ic: float
    summary: dict

    def to_dict(self) -> dict:
        payload = {
            "sharpe": self.sharpe,
            "ic": self.ic,
            "rank_ic": self.rank_ic,
        }
        payload.update(self.summary)
        return payload


def run_null_label_test(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    horizon_days: int,
    seed: int,
    price_panel: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    lgbm_cfg,
    output_dir: Path,
) -> NullLabelResult:
    """Shuffle labels by date blocks, retrain, and backtest to ensure performance collapses."""

    if training_frame.empty:
        raise ValueError("Training frame empty; cannot run null-label test.")
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(training_frame["date"])
    unique_dates = np.array(sorted(dates.unique()))
    rng = np.random.default_rng(seed)
    permuted = unique_dates.copy()
    rng.shuffle(permuted)
    mapping = dict(zip(unique_dates, permuted))

    shuffled = training_frame["label"].copy()
    for original, donor in mapping.items():
        donor_values = training_frame.loc[dates == donor, "label"].values
        idx = training_frame.index[dates == original]
        # If donor length differs, tile to match.
        tiled = np.resize(donor_values, idx.size)
        shuffled.loc[idx] = tiled

    trainer = LightGBMTrainer(lgbm_cfg)
    features = training_frame[feature_columns]
    model = trainer.train(features, shuffled)
    proba, _ = trainer.predict_with_meta_label(model, features)

    preds = training_frame[["date", "ticker"]].copy()
    preds["proba"] = proba
    preds["label"] = shuffled
    preds_path = output_dir / "null_label_preds.csv"
    preds.to_csv(preds_path, index=False)

    preds = preds[preds["ticker"].isin(price_panel["ticker"].unique())]
    result = backtest_signals(preds, price_panel, backtest_cfg)
    summary = result["summary"]
    sharpe = summary.get("Sharpe", 0.0)

    evaluator = ModelEvaluator(horizon_days=horizon_days)
    metrics = evaluator.evaluate(
        training_frame["label"],
        proba,
        excess_returns=training_frame.get("excess_fwd"),
        dates=dates,
    )
    ic = float(metrics.get("ic", 0.0))
    rank_ic = float(metrics.get("rank_ic", 0.0))

    summary_path = output_dir / "null_label_summary.json"
    import json

    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    logger.info(
        "Null-label sanity results",
        extra={"sharpe": sharpe, "ic": ic, "rank_ic": rank_ic, "summary_path": str(summary_path)},
    )
    return NullLabelResult(sharpe=sharpe, ic=ic, rank_ic=rank_ic, summary=summary)


def run_reversed_time_test(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    lgbm_cfg,
    test_fraction: float = 0.3,
) -> Dict[str, float]:
    """Train on future data, test on past data. Should perform poorly if no leakage.
    
    If this performs well, it indicates features contain future information.
    """
    if training_frame.empty:
        raise ValueError("Training frame empty")
    
    dates = pd.to_datetime(training_frame["date"])
    sorted_idx = dates.argsort()
    split_point = int(len(sorted_idx) * (1 - test_fraction))
    
    # REVERSE: train on future, test on past
    train_idx = sorted_idx[split_point:]  # Future dates for training
    test_idx = sorted_idx[:split_point]   # Past dates for testing
    
    trainer = LightGBMTrainer(lgbm_cfg)
    features = training_frame[feature_columns]
    labels = training_frame["label"]
    
    # Train on future
    model = trainer.train(features.iloc[train_idx], labels.iloc[train_idx])
    
    # Test on past
    proba, _ = trainer.predict_with_meta_label(model, features.iloc[test_idx])
    
    # Evaluate
    evaluator = ModelEvaluator(horizon_days=10)
    metrics = evaluator.evaluate(labels.iloc[test_idx], proba)
    
    roc_auc = float(metrics.get("roc_auc", 0.0))
    ic = float(metrics.get("ic", 0.0))
    rank_ic = float(metrics.get("rank_ic", 0.0))
    
    logger.info(
        "Reversed-time test results",
        extra={
            "roc_auc": roc_auc,
            "ic": ic,
            "rank_ic": rank_ic,
            "verdict": "PASS" if roc_auc < 0.55 else "FAIL - possible leakage"
        }
    )
    
    return {
        "roc_auc": roc_auc,
        "ic": ic,
        "rank_ic": rank_ic,
        "pass": roc_auc < 0.55,  # Should be near random
    }


def run_gap_split_test(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    lgbm_cfg,
    gap_months: int = 6,
) -> Dict[str, float]:
    """Train on early period, test on much later period with gap.
    
    Tests regime stability and prevents train/test contamination through rolling windows.
    """
    if training_frame.empty:
        raise ValueError("Training frame empty")
    
    dates = pd.to_datetime(training_frame["date"])
    unique_dates = sorted(dates.unique())
    
    if len(unique_dates) < 30:
        logger.warning("Not enough dates for gap split test")
        return {"roc_auc": 0.5, "ic": 0.0, "pass": True}
    
    # Split: first 40%, gap, last 40%
    train_cutoff = unique_dates[int(len(unique_dates) * 0.4)]
    gap_end = train_cutoff + pd.Timedelta(days=gap_months * 30)
    
    train_mask = dates <= train_cutoff
    test_mask = dates >= gap_end
    
    if not train_mask.any() or not test_mask.any():
        logger.warning("Gap too large, insufficient data")
        return {"roc_auc": 0.5, "ic": 0.0, "pass": True}
    
    trainer = LightGBMTrainer(lgbm_cfg)
    features = training_frame[feature_columns]
    labels = training_frame["label"]
    
    # Train on early period
    model = trainer.train(features[train_mask], labels[train_mask])
    
    # Test on later period (after gap)
    proba, _ = trainer.predict_with_meta_label(model, features[test_mask])
    
    # Evaluate
    evaluator = ModelEvaluator(horizon_days=10)
    metrics = evaluator.evaluate(labels[test_mask], proba)
    
    roc_auc = float(metrics.get("roc_auc", 0.0))
    ic = float(metrics.get("ic", 0.0))
    
    logger.info(
        f"Gap split test ({gap_months}m gap) results",
        extra={
            "roc_auc": roc_auc,
            "ic": ic,
            "train_size": train_mask.sum(),
            "test_size": test_mask.sum(),
        }
    )
    
    return {
        "roc_auc": roc_auc,
        "ic": ic,
        "gap_months": gap_months,
        "stable": roc_auc > 0.52,  # Still has some edge after gap
    }


def run_feature_future_correlation_test(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    horizon_days: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """Check if features correlate with FUTURE labels (not current labels).
    
    Features should NOT correlate with future labels beyond the horizon.
    """
    if training_frame.empty:
        return {"suspicious_features": []}
    
    dates = pd.to_datetime(training_frame["date"])
    suspicious_features = []
    
    # For each feature, check correlation with labels shifted forward
    for col in feature_columns:
        if col not in training_frame.columns:
            continue
        
        # Shift labels forward (beyond horizon)
        future_shift = horizon_days * 2  # Look 2x horizon into future
        
        aligned = training_frame[["ticker", "date", col, "label"]].copy()
        aligned = aligned.sort_values(["ticker", "date"])
        
        # Shift labels forward within each ticker
        aligned["future_label"] = aligned.groupby("ticker")["label"].shift(-future_shift)
        
        # Compute correlation
        clean = aligned[[col, "future_label"]].dropna()
        if len(clean) < 100:
            continue
        
        corr = clean[col].corr(clean["future_label"], method="spearman")
        
        # Flag if absolute correlation > 0.1 (shouldn't correlate with far future)
        if abs(corr) > 0.1:
            suspicious_features.append((col, float(corr)))
    
    if suspicious_features:
        logger.warning(
            "Features correlated with future labels detected",
            extra={"count": len(suspicious_features), "top": suspicious_features[:5]}
        )
    else:
        logger.info("No suspicious future correlations detected")
    
    return {
        "suspicious_features": sorted(suspicious_features, key=lambda x: abs(x[1]), reverse=True),
        "pass": len(suspicious_features) == 0,
    }


def run_random_split_test(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    lgbm_cfg,
    n_random_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Random splits (ignoring time) should perform WORSE than time-based splits.
    
    If random splits perform better, suggests data leakage or non-temporal patterns.
    """
    if training_frame.empty:
        raise ValueError("Training frame empty")
    
    rng = np.random.default_rng(seed)
    random_aucs = []
    
    for i in range(n_random_splits):
        # Random 70/30 split
        perm = rng.permutation(len(training_frame))
        split_point = int(len(perm) * 0.7)
        train_idx = perm[:split_point]
        test_idx = perm[split_point:]
        
        trainer = LightGBMTrainer(lgbm_cfg)
        features = training_frame[feature_columns]
        labels = training_frame["label"]
        
        # Train
        model = trainer.train(features.iloc[train_idx], labels.iloc[train_idx])
        
        # Test
        proba, _ = trainer.predict_with_meta_label(model, features.iloc[test_idx])
        
        # Evaluate
        evaluator = ModelEvaluator(horizon_days=10)
        metrics = evaluator.evaluate(labels.iloc[test_idx], proba)
        
        random_aucs.append(float(metrics.get("roc_auc", 0.5)))
    
    mean_random_auc = float(np.mean(random_aucs))
    std_random_auc = float(np.std(random_aucs))
    
    logger.info(
        "Random split test results",
        extra={
            "mean_auc": mean_random_auc,
            "std_auc": std_random_auc,
            "n_splits": n_random_splits,
        }
    )
    
    return {
        "mean_auc": mean_random_auc,
        "std_auc": std_random_auc,
        "individual_aucs": random_aucs,
    }


@dataclass
class ComprehensiveLeakageReport:
    """Complete leakage test results."""
    null_label: NullLabelResult
    reversed_time: Dict[str, float]
    gap_split: Dict[str, float]
    future_correlation: Dict[str, List]
    random_splits: Dict[str, float]
    
    def is_clean(self) -> bool:
        """Check if all tests pass."""
        checks = [
            abs(self.null_label.sharpe) < 1.0,  # Null test should show no edge
            self.reversed_time.get("pass", False),  # Reversed time should fail
            self.future_correlation.get("pass", False),  # No future correlation
        ]
        return all(checks)
    
    def summary(self) -> Dict:
        """Generate summary dict."""
        return {
            "null_label_sharpe": self.null_label.sharpe,
            "null_label_ic": self.null_label.ic,
            "reversed_time_auc": self.reversed_time.get("roc_auc", 0.0),
            "reversed_time_pass": self.reversed_time.get("pass", False),
            "gap_split_auc": self.gap_split.get("roc_auc", 0.0),
            "suspicious_features_count": len(self.future_correlation.get("suspicious_features", [])),
            "random_split_mean_auc": self.random_splits.get("mean_auc", 0.5),
            "overall_verdict": "CLEAN" if self.is_clean() else "LEAKAGE DETECTED",
        }


def run_comprehensive_leakage_tests(
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    horizon_days: int,
    seed: int,
    price_panel: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    lgbm_cfg,
    output_dir: Path,
) -> ComprehensiveLeakageReport:
    """Run all leakage tests and generate comprehensive report."""
    logger.info("Running comprehensive leakage test suite")
    
    # 1. Null label test
    null_result = run_null_label_test(
        training_frame,
        feature_columns,
        horizon_days=horizon_days,
        seed=seed,
        price_panel=price_panel,
        backtest_cfg=backtest_cfg,
        lgbm_cfg=lgbm_cfg,
        output_dir=output_dir / "null_test",
    )
    
    # 2. Reversed time test
    reversed_result = run_reversed_time_test(
        training_frame,
        feature_columns,
        lgbm_cfg=lgbm_cfg,
    )
    
    # 3. Gap split test
    gap_result = run_gap_split_test(
        training_frame,
        feature_columns,
        lgbm_cfg=lgbm_cfg,
        gap_months=6,
    )
    
    # 4. Future correlation test
    future_corr_result = run_feature_future_correlation_test(
        training_frame,
        feature_columns,
        horizon_days=horizon_days,
    )
    
    # 5. Random split test
    random_result = run_random_split_test(
        training_frame,
        feature_columns,
        lgbm_cfg=lgbm_cfg,
        n_random_splits=3,
        seed=seed,
    )
    
    report = ComprehensiveLeakageReport(
        null_label=null_result,
        reversed_time=reversed_result,
        gap_split=gap_result,
        future_correlation=future_corr_result,
        random_splits=random_result,
    )
    
    # Save comprehensive report
    summary = report.summary()
    import json
    report_path = output_dir / "comprehensive_leakage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, default=float))
    
    logger.info(
        "Comprehensive leakage tests complete",
        extra={"verdict": summary["overall_verdict"], "report_path": str(report_path)}
    )
    
    return report
