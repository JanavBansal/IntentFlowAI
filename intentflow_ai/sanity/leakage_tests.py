"""Leakage detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

    def _compute(group: pd.DataFrame) -> pd.Series:
        shifted = group[price_col].shift(-horizon_days)
        return shifted / group[price_col] - 1.0

    expected = (
        prices.groupby(ticker_col, group_keys=False)
        .apply(_compute)
        .rename("expected_fwd")
    )
    aligned = frame[[date_col, ticker_col, fwd_ret_col]].copy()
    aligned = aligned.merge(
        expected.reset_index(name="expected_fwd"),
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
