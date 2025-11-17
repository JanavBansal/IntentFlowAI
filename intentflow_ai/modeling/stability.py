"""Model parameter optimization for stability over peak performance.

Focuses on:
- Consistent out-of-sample performance across time
- Low variance in cross-validation
- Robust to parameter perturbations
- Preferring underfit over overfit
- Stable feature importance rankings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from intentflow_ai.config.settings import LightGBMConfig
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.modeling.evaluation import ModelEvaluator
from intentflow_ai.utils.logging import get_logger
from intentflow_ai.utils.splits import purged_time_series_splits

logger = get_logger(__name__)


@dataclass
class StabilityConfig:
    """Configuration for stability-focused optimization."""
    
    # Objective weights (should sum to 1.0)
    mean_performance_weight: float = 0.3  # Mean CV score
    stability_weight: float = 0.5  # Low variance across folds
    robustness_weight: float = 0.2  # Insensitive to parameter changes
    
    # Metrics to optimize
    primary_metric: str = "rank_ic"  # "rank_ic", "sharpe", "hit_rate"
    minimize_variance: bool = True
    
    # Cross-validation
    n_cv_folds: int = 5
    embargo_days: int = 10
    
    # Robustness testing
    test_parameter_robustness: bool = True
    parameter_perturbation_pct: float = 0.10  # ±10% parameter perturbation
    
    # Conservative parameter constraints
    max_depth: int = 7  # Limit tree depth
    max_num_leaves: int = 63
    min_learning_rate: float = 0.01
    max_learning_rate: float = 0.10
    min_n_estimators: int = 100
    max_n_estimators: int = 1000
    min_regularization: float = 0.1  # Force some regularization


@dataclass
class StabilityOptimizer:
    """Optimize model parameters for stability and robustness."""
    
    cfg: StabilityConfig = field(default_factory=StabilityConfig)
    
    def optimize(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        *,
        param_grid: Optional[Dict] = None,
    ) -> Tuple[LightGBMConfig, Dict[str, object]]:
        """Find stable parameters via cross-validation.
        
        Args:
            features: Feature matrix
            labels: Target labels
            dates: Date series for time-based splits
            param_grid: Optional parameter grid to search
            
        Returns:
            Tuple of (best_config, optimization_results)
        """
        if param_grid is None:
            param_grid = self._get_conservative_param_grid()
        
        logger.info(
            "Starting stability optimization",
            extra={
                "n_candidates": len(list(ParameterGrid(param_grid))),
                "cv_folds": self.cfg.n_cv_folds
            }
        )
        
        # Prepare CV splits
        cv_frame = pd.DataFrame({"date": dates})
        try:
            cv_splits = purged_time_series_splits(
                cv_frame,
                n_splits=self.cfg.n_cv_folds,
                embargo_days=self.cfg.embargo_days,
                horizon_days=self.cfg.embargo_days,
            )
        except Exception as exc:
            logger.warning(f"Failed to create purged splits: {exc}, using simple time splits")
            # Fallback: simple chronological splits
            cv_splits = self._simple_time_splits(dates, self.cfg.n_cv_folds)
        
        # Grid search with stability scoring
        results = []
        for params in ParameterGrid(param_grid):
            result = self._evaluate_parameter_set(
                params=params,
                features=features,
                labels=labels,
                cv_splits=cv_splits,
            )
            results.append(result)
        
        # Select best based on stability score
        best_result = max(results, key=lambda x: x["stability_score"])
        best_params = best_result["params"]
        
        # Create final config
        best_config = LightGBMConfig(**best_params)
        
        logger.info(
            "Stability optimization complete",
            extra={
                "best_stability_score": best_result["stability_score"],
                "mean_performance": best_result["mean_performance"],
                "cv_std": best_result["cv_std"],
            }
        )
        
        optimization_results = {
            "best_params": best_params,
            "best_stability_score": best_result["stability_score"],
            "all_results": results,
            "cv_splits_used": len(cv_splits),
        }
        
        return best_config, optimization_results
    
    def _evaluate_parameter_set(
        self,
        params: Dict,
        features: pd.DataFrame,
        labels: pd.Series,
        cv_splits: List[Tuple],
    ) -> Dict[str, object]:
        """Evaluate a single parameter set across CV folds."""
        config = LightGBMConfig(**params)
        trainer = LightGBMTrainer(config)
        
        fold_scores = []
        feature_importances = []
        
        for train_mask, test_mask in cv_splits:
            if not train_mask.any() or not test_mask.any():
                continue
            
            try:
                # Train model
                model = trainer.train(features[train_mask], labels[train_mask])
                
                # Predict on test fold
                proba, _ = trainer.predict_with_meta_label(model, features[test_mask])
                
                # Evaluate
                evaluator = ModelEvaluator(horizon_days=10)
                metrics = evaluator.evaluate(labels[test_mask], proba)
                
                score = metrics.get(self.cfg.primary_metric, 0.0)
                if pd.isna(score):
                    score = 0.0
                fold_scores.append(float(score))
                
                # Track feature importance stability
                importances = trainer.feature_importance(model)
                feature_importances.append(importances)
                
            except Exception as exc:
                logger.debug(f"CV fold failed: {exc}")
                fold_scores.append(0.0)
        
        if not fold_scores:
            return {
                "params": params,
                "mean_performance": 0.0,
                "cv_std": np.inf,
                "stability_score": 0.0,
                "fold_scores": [],
            }
        
        # Compute stability metrics
        mean_performance = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))
        
        # Feature importance stability (low variance = stable)
        importance_stability = self._compute_importance_stability(feature_importances)
        
        # Parameter robustness (optional)
        robustness_score = 1.0  # Placeholder
        if self.cfg.test_parameter_robustness:
            robustness_score = self._test_parameter_robustness(
                params, features, labels, cv_splits
            )
        
        # Combined stability score
        stability_score = (
            self.cfg.mean_performance_weight * max(mean_performance, 0) +
            self.cfg.stability_weight * (1.0 / (1.0 + cv_std)) +
            self.cfg.robustness_weight * robustness_score
        )
        
        return {
            "params": params,
            "mean_performance": mean_performance,
            "cv_std": cv_std,
            "importance_stability": importance_stability,
            "robustness_score": robustness_score,
            "stability_score": stability_score,
            "fold_scores": fold_scores,
        }
    
    def _get_conservative_param_grid(self) -> Dict[str, List]:
        """Generate conservative parameter grid.
        
        Focuses on regularization and simplicity to avoid overfitting.
        """
        return {
            "learning_rate": [0.01, 0.03, 0.05],
            "n_estimators": [300, 500, 700],
            "max_depth": [5, 7, -1],
            "num_leaves": [31, 63],
            "feature_fraction": [0.6, 0.7, 0.8],
            "subsample": [0.7, 0.8, 0.9],
            "subsample_freq": [1],
            "reg_lambda": [1.0, 2.0, 5.0],  # L2 regularization
            "reg_alpha": [0.0, 0.5, 1.0],  # L1 regularization
            "min_child_samples": [20, 50],
        }
    
    def _simple_time_splits(self, dates: pd.Series, n_splits: int) -> List[Tuple]:
        """Create simple chronological splits as fallback."""
        sorted_dates = dates.sort_values()
        unique_dates = sorted_dates.unique()
        
        if len(unique_dates) < n_splits:
            return []
        
        fold_size = len(sorted_dates) // n_splits
        splits = []
        
        for i in range(n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else len(sorted_dates)
            
            train_mask = sorted_dates.index < sorted_dates.index[test_start]
            test_mask = (sorted_dates.index >= sorted_dates.index[test_start]) & \
                       (sorted_dates.index < sorted_dates.index[min(test_end, len(sorted_dates) - 1)])
            
            if train_mask.any() and test_mask.any():
                splits.append((train_mask, test_mask))
        
        return splits
    
    def _compute_importance_stability(self, importance_list: List[Dict[str, float]]) -> float:
        """Compute stability of feature importances across folds.
        
        Returns a score where 1.0 = perfectly stable, 0.0 = highly unstable.
        """
        if len(importance_list) < 2:
            return 1.0
        
        # Convert to DataFrame for analysis
        imp_df = pd.DataFrame(importance_list).fillna(0)
        
        if imp_df.empty:
            return 1.0
        
        # Compute coefficient of variation for each feature
        mean_imp = imp_df.mean(axis=0)
        std_imp = imp_df.std(axis=0)
        
        cv = std_imp / (mean_imp + 1e-9)
        
        # Average CV (lower = more stable)
        avg_cv = cv.mean()
        
        # Transform to 0-1 scale (lower CV = higher stability score)
        stability = 1.0 / (1.0 + avg_cv)
        
        return float(stability)
    
    def _test_parameter_robustness(
        self,
        params: Dict,
        features: pd.DataFrame,
        labels: pd.Series,
        cv_splits: List[Tuple],
    ) -> float:
        """Test how sensitive performance is to parameter perturbations.
        
        Returns robustness score where 1.0 = performance stable under perturbations.
        """
        # Get baseline performance
        baseline_result = self._evaluate_parameter_set(params, features, labels, cv_splits)
        baseline_score = baseline_result["mean_performance"]
        
        # Perturb numeric parameters
        perturbed_scores = []
        numeric_params = {k: v for k, v in params.items() if isinstance(v, (int, float))}
        
        for param_name, base_value in numeric_params.items():
            if param_name in ["random_state", "verbose", "n_jobs"]:
                continue
            
            # Try small perturbations
            for direction in [-1, 1]:
                perturbed_value = base_value * (1 + direction * self.cfg.parameter_perturbation_pct)
                
                # Ensure valid range
                if param_name == "learning_rate":
                    perturbed_value = np.clip(perturbed_value, 0.001, 0.5)
                elif param_name == "n_estimators":
                    perturbed_value = int(np.clip(perturbed_value, 10, 2000))
                
                perturbed_params = params.copy()
                perturbed_params[param_name] = perturbed_value
                
                try:
                    result = self._evaluate_parameter_set(
                        perturbed_params, features, labels, [cv_splits[0]]  # Only first fold for speed
                    )
                    perturbed_scores.append(result["mean_performance"])
                except Exception:
                    continue
        
        if not perturbed_scores:
            return 0.5  # Neutral if can't test
        
        # Robustness = inverse of relative performance variance
        score_variance = np.std(perturbed_scores)
        robustness = 1.0 / (1.0 + score_variance / (abs(baseline_score) + 1e-9))
        
        return float(robustness)


def compare_to_baseline(
    optimized_config: LightGBMConfig,
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    baseline_type: str = "linear",
) -> Dict[str, float]:
    """Compare optimized model to simple baseline.
    
    Args:
        optimized_config: Optimized LightGBM config
        features: Feature matrix
        labels: Labels
        baseline_type: "linear" or "trivial" (always predict mean)
        
    Returns:
        Dict with comparison metrics
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    # Train optimized model
    trainer = LightGBMTrainer(optimized_config)
    model = trainer.train(features, labels)
    opt_proba, _ = trainer.predict_with_meta_label(model, features)
    
    # Baseline model
    if baseline_type == "linear":
        baseline = LogisticRegression(max_iter=1000, random_state=42)
        baseline.fit(features.fillna(0), labels)
        baseline_proba = baseline.predict_proba(features.fillna(0))[:, 1]
    else:
        # Trivial: always predict label mean
        baseline_proba = np.full(len(labels), labels.mean())
    
    # Compare
    opt_auc = roc_auc_score(labels, opt_proba)
    baseline_auc = roc_auc_score(labels, baseline_proba)
    
    # IC comparison - ensure everything is Series with matching indices
    labels_series = pd.Series(labels) if not isinstance(labels, pd.Series) else labels
    opt_proba_series = pd.Series(opt_proba) if not isinstance(opt_proba, pd.Series) else opt_proba
    baseline_proba_series = pd.Series(baseline_proba)
    
    # Reset indices to ensure alignment
    labels_series = labels_series.reset_index(drop=True)
    opt_proba_series = opt_proba_series.reset_index(drop=True)
    baseline_proba_series = baseline_proba_series.reset_index(drop=True)
    
    opt_ic = labels_series.corr(opt_proba_series, method="spearman")
    baseline_ic = labels_series.corr(baseline_proba_series, method="spearman")
    
    return {
        "optimized_auc": float(opt_auc),
        "baseline_auc": float(baseline_auc),
        "auc_improvement": float(opt_auc - baseline_auc),
        "optimized_ic": float(opt_ic),
        "baseline_ic": float(baseline_ic),
        "ic_improvement": float(opt_ic - baseline_ic),
        "baseline_type": baseline_type,
    }


def generate_stability_report(
    optimization_results: Dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate stability optimization report.
    
    Args:
        optimization_results: Output from StabilityOptimizer.optimize()
        output_path: Optional path to save markdown report
        
    Returns:
        Report as markdown string
    """
    lines = ["# Model Stability Optimization Report\n"]
    
    best_params = optimization_results.get("best_params", {})
    best_score = optimization_results.get("best_stability_score", 0.0)
    
    lines.append("## Best Parameters\n")
    lines.append("```python")
    for key, value in best_params.items():
        lines.append(f"{key}: {value}")
    lines.append("```\n")
    
    lines.append(f"**Stability Score**: {best_score:.4f}\n")
    
    # All results summary
    all_results = optimization_results.get("all_results", [])
    if all_results:
        lines.append("## Parameter Search Summary\n")
        lines.append(f"- **Configurations tested**: {len(all_results)}")
        
        stability_scores = [r["stability_score"] for r in all_results]
        lines.append(f"- **Best stability score**: {max(stability_scores):.4f}")
        lines.append(f"- **Worst stability score**: {min(stability_scores):.4f}")
        lines.append(f"- **Mean stability score**: {np.mean(stability_scores):.4f}\n")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Stability report saved", extra={"path": output_path})
    
    return report

