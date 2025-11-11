"""Visualization helpers for model diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_classification_plots(
    y_true,
    y_score,
    hit_curve: Iterable[dict] | None,
    output_dir: Path,
) -> Dict[str, Path]:
    """Persist ROC, PR, and hit-rate curve plots and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("ROC Curve")
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    paths["roc_curve"] = roc_path

    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Precision-Recall Curve")
    pr_path = output_dir / "precision_recall_curve.png"
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)
    paths["precision_recall_curve"] = pr_path

    curve = list(hit_curve or [])
    if curve:
        thresholds = [point["threshold"] for point in curve]
        hit_rates = [point["hit_rate"] for point in curve]
        coverage = [point["coverage"] for point in curve]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(thresholds, hit_rates, marker="o", label="Hit rate", color="tab:blue")
        ax1.set_xlabel("Score threshold")
        ax1.set_ylabel("Hit rate", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(thresholds, coverage, marker="x", label="Coverage", color="tab:orange")
        ax2.set_ylabel("Coverage", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.set_title("Hit-rate vs Threshold")

        hit_curve_path = output_dir / "hit_rate_curve.png"
        fig.tight_layout()
        fig.savefig(hit_curve_path, bbox_inches="tight")
        plt.close(fig)
        paths["hit_rate_curve"] = hit_curve_path

    return paths
