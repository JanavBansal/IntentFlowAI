"""
Probability Calibration Module

Implements probability calibration for ensemble model outputs.
Ensures predicted probabilities reflect true outcome frequencies.

Methods:
- Platt Scaling (Sigmoid): Parametric, good for small datasets
- Isotonic Regression: Non-parametric, flexible, needs more data

Usage:
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(model, X_calib, y_calib)
    calibrated_probs = calibrator.predict_proba(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class CalibrationMethod(Enum):
    """Calibration method options."""
    PLATT = "sigmoid"       # Platt Scaling (LogisticRegression)
    ISOTONIC = "isotonic"   # Isotonic Regression
    VENN_ABERS = "venn"     # Venn-ABERS (if available)


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""
    brier_score: float           # Lower is better (0 = perfect)
    expected_calibration_error: float  # ECE
    max_calibration_error: float  # MCE
    reliability_curve: Tuple[np.ndarray, np.ndarray]  # (fraction_of_positives, mean_predicted_value)
    
    def is_well_calibrated(self, threshold: float = 0.25) -> bool:
        """Check if calibration is acceptable."""
        return self.brier_score < threshold


@dataclass
class ProbabilityCalibrator:
    """
    Calibrate model probabilities using held-out calibration data.
    
    Transforms raw classifier outputs into calibrated probabilities
    that more accurately reflect true outcome frequencies.
    """
    
    method: str = "isotonic"  # "sigmoid" or "isotonic"
    n_bins: int = 10          # Number of bins for reliability diagram
    _calibrator: Optional[object] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    _metrics: Optional[CalibrationMetrics] = field(default=None, init=False)
    
    def fit(
        self,
        base_model: object,
        X_calib: Union[np.ndarray, pd.DataFrame],
        y_calib: Union[np.ndarray, pd.Series]
    ) -> "ProbabilityCalibrator":
        """
        Fit calibrator on held-out calibration set.
        
        Args:
            base_model: Pre-trained classifier with predict_proba method
            X_calib: Calibration features
            y_calib: Calibration labels
            
        Returns:
            self (fitted calibrator)
        """
        if hasattr(X_calib, 'values'):
            X_calib = X_calib.values
        if hasattr(y_calib, 'values'):
            y_calib = y_calib.values
        
        # Get raw probabilities from base model
        try:
            raw_probs = base_model.predict_proba(X_calib)
            if raw_probs.ndim == 2:
                raw_probs = raw_probs[:, 1]  # Get positive class probs
        except Exception as e:
            logger.error(f"Failed to get base model predictions: {e}")
            raise
        
        # Fit calibrator based on method
        if self.method == "sigmoid":
            self._fit_platt(raw_probs, y_calib)
        elif self.method == "isotonic":
            self._fit_isotonic(raw_probs, y_calib)
        else:
            logger.warning(f"Unknown method '{self.method}', defaulting to isotonic")
            self._fit_isotonic(raw_probs, y_calib)
        
        self._is_fitted = True
        
        # Compute calibration metrics
        calibrated_probs = self._calibrator.predict(raw_probs.reshape(-1, 1)).flatten() \
            if self.method == "sigmoid" else self._calibrator.predict(raw_probs)
        self._compute_metrics(y_calib, calibrated_probs, raw_probs)
        
        logger.info(
            f"Calibrator fitted: method={self.method}, "
            f"brier_score={self._metrics.brier_score:.4f}"
        )
        
        return self
    
    def _fit_platt(self, raw_probs: np.ndarray, y: np.ndarray) -> None:
        """Fit Platt Scaling (sigmoid) calibration."""
        self._calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
        self._calibrator.fit(raw_probs.reshape(-1, 1), y)
    
    def _fit_isotonic(self, raw_probs: np.ndarray, y: np.ndarray) -> None:
        """Fit Isotonic Regression calibration."""
        self._calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        self._calibrator.fit(raw_probs, y)
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        base_model: Optional[object] = None,
        raw_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return calibrated probabilities.
        
        Args:
            X: Features (if base_model provided)
            base_model: Pre-trained classifier (optional if raw_probs provided)
            raw_probs: Raw probabilities (optional if base_model provided)
            
        Returns:
            Calibrated probability array
        """
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        # Get raw probabilities
        if raw_probs is None:
            if base_model is None:
                raise ValueError("Either base_model or raw_probs must be provided")
            raw_probs = base_model.predict_proba(X)
            if raw_probs.ndim == 2:
                raw_probs = raw_probs[:, 1]
        
        # Calibrate
        if self.method == "sigmoid":
            calibrated = self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:
            calibrated = self._calibrator.predict(raw_probs)
        
        return calibrated
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_calibrated: np.ndarray,
        y_raw: np.ndarray
    ) -> None:
        """Compute calibration metrics."""
        # Brier Score
        brier = np.mean((y_calibrated - y_true) ** 2)
        
        # Reliability curve
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true, y_calibrated, n_bins=self.n_bins, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        bin_counts = np.zeros(self.n_bins)
        bin_errors = np.zeros(self.n_bins)
        
        for i in range(len(y_calibrated)):
            bin_idx = min(int(y_calibrated[i] * self.n_bins), self.n_bins - 1)
            bin_counts[bin_idx] += 1
            bin_errors[bin_idx] += abs(y_calibrated[i] - y_true[i])
        
        # Avoid division by zero
        valid_bins = bin_counts > 0
        ece = np.sum(bin_errors[valid_bins]) / len(y_calibrated) if len(y_calibrated) > 0 else 0
        
        # Max Calibration Error
        if len(fraction_of_positives) > 0 and len(mean_predicted) > 0:
            mce = np.max(np.abs(fraction_of_positives - mean_predicted))
        else:
            mce = 0.0
        
        self._metrics = CalibrationMetrics(
            brier_score=brier,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            reliability_curve=(fraction_of_positives, mean_predicted)
        )
    
    def get_metrics(self) -> Optional[CalibrationMetrics]:
        """Get calibration metrics."""
        return self._metrics
    
    def plot_reliability_diagram(self, save_path: Optional[str] = None) -> None:
        """Plot reliability diagram (calibration curve)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        if self._metrics is None:
            logger.warning("No metrics available. Fit calibrator first.")
            return
        
        frac_pos, mean_pred = self._metrics.reliability_curve
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Actual calibration curve
        ax.plot(mean_pred, frac_pos, 's-', label=f'Calibrated ({self.method})')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Reliability Diagram (Brier={self._metrics.brier_score:.4f})')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved reliability diagram to {save_path}")
        
        plt.close()


def compare_calibration_methods(
    base_model: object,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, CalibrationMetrics]:
    """
    Compare Platt Scaling vs Isotonic Regression.
    
    Returns:
        Dict mapping method name to CalibrationMetrics
    """
    results = {}
    
    for method in ["sigmoid", "isotonic"]:
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(base_model, X_calib, y_calib)
        
        # Evaluate on test set
        raw_probs = base_model.predict_proba(X_test)
        if raw_probs.ndim == 2:
            raw_probs = raw_probs[:, 1]
        
        calibrated_probs = calibrator.predict_proba(X_test, raw_probs=raw_probs)
        
        # Compute test metrics
        brier = np.mean((calibrated_probs - y_test) ** 2)
        
        frac_pos, mean_pred = calibration_curve(
            y_test, calibrated_probs, n_bins=10, strategy='uniform'
        )
        
        ece = np.mean(np.abs(frac_pos - mean_pred)) if len(frac_pos) > 0 else 0
        mce = np.max(np.abs(frac_pos - mean_pred)) if len(frac_pos) > 0 else 0
        
        results[method] = CalibrationMetrics(
            brier_score=brier,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            reliability_curve=(frac_pos, mean_pred)
        )
        
        logger.info(f"{method}: Brier={brier:.4f}, ECE={ece:.4f}")
    
    return results


def split_calibration_set(
    X: np.ndarray,
    y: np.ndarray,
    calib_fraction: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and calibration sets.
    
    Args:
        X: Features
        y: Labels
        calib_fraction: Fraction for calibration (default 15%)
        
    Returns:
        X_train, y_train, X_calib, y_calib
    """
    n = len(X)
    n_calib = int(n * calib_fraction)
    
    # Use last portion for calibration (time-series appropriate)
    X_train = X[:-n_calib]
    y_train = y[:-n_calib]
    X_calib = X[-n_calib:]
    y_calib = y[-n_calib:]
    
    return X_train, y_train, X_calib, y_calib
