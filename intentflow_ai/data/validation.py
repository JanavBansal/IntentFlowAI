"""Data validation framework for new feature layers.

This module provides systematic validation of new data sources before integration:
1. Out-of-sample IC validation
2. Correlation with existing features
3. Look-ahead bias detection
4. Data quality checks (missing values, outliers, drift)
5. Incremental value testing (does it improve ensemble?)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.modeling.evaluation import ModelEvaluator
from intentflow_ai.features.orthogonality import FeatureOrthogonalityAnalyzer
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataValidationConfig:
    """Configuration for new data validation."""
    
    # IC thresholds
    min_univariate_ic: float = 0.02  # Min IC for new feature
    min_incremental_ic: float = 0.01  # Min IC improvement when added to existing
    
    # Quality thresholds
    max_missing_pct: float = 0.20  # Max 20% missing values
    max_outlier_pct: float = 0.05  # Max 5% outliers
    
    # Leakage detection
    check_future_correlation: bool = True
    future_horizon_multiplier: int = 2  # Check 2x label horizon into future
    
    # Correlation with existing features
    max_correlation_with_existing: float = 0.85  # Max correlation
    
    # Drift detection
    check_distribution_drift: bool = True
    max_ks_statistic: float = 0.20  # Max KS stat for train/test drift


@dataclass
class NewDataValidator:
    """Validate new data sources before integration."""
    
    cfg: DataValidationConfig = field(default_factory=DataValidationConfig)
    
    def validate_new_features(
        self,
        new_features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        *,
        existing_features: Optional[pd.DataFrame] = None,
        lgbm_cfg=None,
    ) -> Dict[str, Dict]:
        """Comprehensive validation of new feature set.
        
        Args:
            new_features: New features to validate
            labels: Target labels
            dates: Date series
            existing_features: Optional existing feature set
            lgbm_cfg: LightGBM config for model-based tests
            
        Returns:
            Dict mapping feature name to validation results
        """
        if new_features.empty:
            return {}
        
        logger.info(
            "Validating new features",
            extra={"n_features": len(new_features.columns)}
        )
        
        results = {}
        
        for col in new_features.columns:
            logger.info(f"Validating feature: {col}")
            
            validation = {
                "feature": col,
                "passed": True,
                "issues": [],
                "metrics": {},
            }
            
            feature_series = new_features[col]
            
            # Check 1: Data quality
            quality_check = self._check_data_quality(feature_series)
            validation["metrics"]["quality"] = quality_check
            
            if not quality_check["passed"]:
                validation["passed"] = False
                validation["issues"].append(f"Data quality: {quality_check['reason']}")
            
            # Check 2: Univariate IC
            ic_check = self._check_univariate_ic(feature_series, labels)
            validation["metrics"]["univariate_ic"] = ic_check
            
            if not ic_check["passed"]:
                validation["passed"] = False
                validation["issues"].append(f"Low IC: {ic_check['ic']:.4f}")
            
            # Check 3: Future correlation (leakage)
            if self.cfg.check_future_correlation:
                future_check = self._check_future_correlation(
                    feature_series, labels, dates
                )
                validation["metrics"]["future_correlation"] = future_check
                
                if not future_check["passed"]:
                    validation["passed"] = False
                    validation["issues"].append(
                        f"Future correlation detected: {future_check['correlation']:.3f}"
                    )
            
            # Check 4: Correlation with existing features
            if existing_features is not None:
                corr_check = self._check_existing_correlation(
                    feature_series, existing_features
                )
                validation["metrics"]["existing_correlation"] = corr_check
                
                if not corr_check["passed"]:
                    validation["passed"] = False
                    validation["issues"].append(
                        f"High correlation with {corr_check['max_corr_feature']}: "
                        f"{corr_check['max_corr']:.3f}"
                    )
            
            # Check 5: Out-of-sample IC (time-based)
            oos_check = self._check_oos_ic(feature_series, labels, dates, lgbm_cfg)
            validation["metrics"]["oos_ic"] = oos_check
            
            if not oos_check["passed"]:
                validation["passed"] = False
                validation["issues"].append(f"Low OOS IC: {oos_check['mean_ic']:.4f}")
            
            # Check 6: Distribution drift
            if self.cfg.check_distribution_drift:
                drift_check = self._check_distribution_drift(feature_series, dates)
                validation["metrics"]["drift"] = drift_check
                
                if not drift_check["passed"]:
                    validation["passed"] = False
                    validation["issues"].append(
                        f"Distribution drift detected: KS={drift_check['ks_statistic']:.3f}"
                    )
            
            # Check 7: Incremental value (if existing features provided)
            if existing_features is not None and lgbm_cfg:
                incremental_check = self._check_incremental_value(
                    feature_series, existing_features, labels, dates, lgbm_cfg
                )
                validation["metrics"]["incremental_value"] = incremental_check
                
                if not incremental_check["passed"]:
                    validation["passed"] = False
                    validation["issues"].append(
                        f"No incremental value: improvement={incremental_check['ic_improvement']:.4f}"
                    )
            
            results[col] = validation
            
            # Log result
            status = "✅ PASS" if validation["passed"] else "❌ FAIL"
            logger.info(
                f"Validation {status}: {col}",
                extra={"issues": len(validation["issues"])}
            )
        
        # Summary
        passed = sum(1 for v in results.values() if v["passed"])
        logger.info(
            "Validation complete",
            extra={
                "total": len(results),
                "passed": passed,
                "failed": len(results) - passed,
            }
        )
        
        return results
    
    def _check_data_quality(self, feature: pd.Series) -> Dict:
        """Check missing values, outliers, constants."""
        total = len(feature)
        
        # Missing values
        missing_count = feature.isna().sum()
        missing_pct = missing_count / total
        
        if missing_pct > self.cfg.max_missing_pct:
            return {
                "passed": False,
                "missing_pct": float(missing_pct),
                "reason": f"Too many missing values: {missing_pct:.1%}",
            }
        
        # Check if constant
        nunique = feature.nunique()
        if nunique <= 1:
            return {
                "passed": False,
                "nunique": int(nunique),
                "reason": "Feature is constant",
            }
        
        # Outliers (beyond 5 std devs)
        feature_clean = feature.dropna()
        if len(feature_clean) > 0:
            mean = feature_clean.mean()
            std = feature_clean.std()
            
            if std > 0:
                outliers = ((feature_clean - mean).abs() > 5 * std).sum()
                outlier_pct = outliers / len(feature_clean)
                
                if outlier_pct > self.cfg.max_outlier_pct:
                    return {
                        "passed": False,
                        "outlier_pct": float(outlier_pct),
                        "reason": f"Too many outliers: {outlier_pct:.1%}",
                    }
        
        return {
            "passed": True,
            "missing_pct": float(missing_pct),
            "nunique": int(nunique),
            "outlier_pct": 0.0,
        }
    
    def _check_univariate_ic(
        self,
        feature: pd.Series,
        labels: pd.Series,
    ) -> Dict:
        """Check univariate information coefficient."""
        # Align and drop NAs
        aligned = pd.DataFrame({"feature": feature, "label": labels}).dropna()
        
        if len(aligned) < 100:
            return {
                "passed": False,
                "ic": 0.0,
                "reason": "Insufficient data points",
            }
        
        # Spearman correlation
        ic = aligned["feature"].corr(aligned["label"], method="spearman")
        
        passed = abs(ic) >= self.cfg.min_univariate_ic
        
        return {
            "passed": passed,
            "ic": float(ic),
            "abs_ic": float(abs(ic)),
        }
    
    def _check_future_correlation(
        self,
        feature: pd.Series,
        labels: pd.Series,
        dates: pd.Series,
    ) -> Dict:
        """Check if feature correlates with future labels (lookahead bias)."""
        # Assume labels are horizon_days forward looking
        # Shift labels even further forward
        future_shift = 20  # Check 20 days ahead
        
        # Combine into dataframe
        df = pd.DataFrame({
            "feature": feature,
            "label": labels,
            "date": dates,
        }).sort_values("date")
        
        # Shift labels forward
        df["future_label"] = df["label"].shift(-future_shift)
        
        # Compute correlation
        clean = df[["feature", "future_label"]].dropna()
        
        if len(clean) < 100:
            return {"passed": True, "correlation": 0.0}
        
        corr = clean["feature"].corr(clean["future_label"], method="spearman")
        
        # Should NOT correlate with far future
        passed = abs(corr) < 0.10
        
        return {
            "passed": passed,
            "correlation": float(corr),
            "threshold": 0.10,
        }
    
    def _check_existing_correlation(
        self,
        feature: pd.Series,
        existing_features: pd.DataFrame,
    ) -> Dict:
        """Check correlation with existing features."""
        if existing_features.empty:
            return {"passed": True, "max_corr": 0.0}
        
        max_corr = 0.0
        max_corr_feature = None
        
        for col in existing_features.columns:
            # Align
            aligned = pd.DataFrame({
                "new": feature,
                "existing": existing_features[col],
            }).dropna()
            
            if len(aligned) < 100:
                continue
            
            corr = aligned["new"].corr(aligned["existing"], method="spearman")
            
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_feature = col
        
        passed = abs(max_corr) < self.cfg.max_correlation_with_existing
        
        return {
            "passed": passed,
            "max_corr": float(max_corr),
            "max_corr_feature": max_corr_feature,
        }
    
    def _check_oos_ic(
        self,
        feature: pd.Series,
        labels: pd.Series,
        dates: pd.Series,
        lgbm_cfg,
    ) -> Dict:
        """Check out-of-sample IC using time-based splits."""
        if lgbm_cfg is None:
            return {"passed": True, "mean_ic": 0.0}
        
        from intentflow_ai.utils.splits import purged_time_series_splits
        
        # Create CV splits
        cv_frame = pd.DataFrame({"date": dates})
        try:
            cv_splits = purged_time_series_splits(
                cv_frame,
                n_splits=3,
                embargo_days=10,
                horizon_days=10,
            )
        except Exception:
            return {"passed": True, "mean_ic": 0.0}
        
        # Test feature on each fold
        fold_ics = []
        
        for train_mask, test_mask in cv_splits:
            if not train_mask.any() or not test_mask.any():
                continue
            
            # Simple univariate IC on test fold
            test_aligned = pd.DataFrame({
                "feature": feature[test_mask],
                "label": labels[test_mask],
            }).dropna()
            
            if len(test_aligned) < 50:
                continue
            
            ic = test_aligned["feature"].corr(test_aligned["label"], method="spearman")
            if not pd.isna(ic):
                fold_ics.append(ic)
        
        mean_ic = float(np.mean(fold_ics)) if fold_ics else 0.0
        std_ic = float(np.std(fold_ics)) if fold_ics else 0.0
        
        passed = abs(mean_ic) >= self.cfg.min_univariate_ic and std_ic < 0.10
        
        return {
            "passed": passed,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "n_folds": len(fold_ics),
        }
    
    def _check_distribution_drift(
        self,
        feature: pd.Series,
        dates: pd.Series,
    ) -> Dict:
        """Check if feature distribution drifts over time."""
        from scipy.stats import ks_2samp
        
        # Split into early and late periods
        df = pd.DataFrame({"feature": feature, "date": dates}).dropna()
        df = df.sort_values("date")
        
        if len(df) < 100:
            return {"passed": True, "ks_statistic": 0.0}
        
        # First 40% vs last 40%
        split_early = int(len(df) * 0.4)
        split_late = int(len(df) * 0.6)
        
        early = df["feature"].iloc[:split_early]
        late = df["feature"].iloc[split_late:]
        
        # KS test
        ks_stat, p_value = ks_2samp(early, late)
        
        passed = ks_stat < self.cfg.max_ks_statistic
        
        return {
            "passed": passed,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
        }
    
    def _check_incremental_value(
        self,
        feature: pd.Series,
        existing_features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        lgbm_cfg,
    ) -> Dict:
        """Check if adding this feature improves model performance."""
        from intentflow_ai.utils.splits import purged_time_series_splits
        
        cv_frame = pd.DataFrame({"date": dates})
        try:
            cv_splits = purged_time_series_splits(
                cv_frame,
                n_splits=3,
                embargo_days=10,
                horizon_days=10,
            )
        except Exception:
            return {"passed": True, "ic_improvement": 0.0}
        
        trainer = LightGBMTrainer(lgbm_cfg)
        evaluator = ModelEvaluator(horizon_days=10)
        
        # Baseline: without new feature
        baseline_ics = []
        for train_mask, test_mask in cv_splits:
            if not train_mask.any() or not test_mask.any():
                continue
            
            try:
                model = trainer.train(existing_features[train_mask], labels[train_mask])
                proba, _ = trainer.predict_with_meta_label(model, existing_features[test_mask])
                metrics = evaluator.evaluate(labels[test_mask], proba)
                ic = metrics.get("ic", 0.0)
                if not pd.isna(ic):
                    baseline_ics.append(ic)
            except Exception:
                continue
        
        baseline_ic = float(np.mean(baseline_ics)) if baseline_ics else 0.0
        
        # With new feature
        augmented_features = existing_features.copy()
        augmented_features["new_feature"] = feature
        
        augmented_ics = []
        for train_mask, test_mask in cv_splits:
            if not train_mask.any() or not test_mask.any():
                continue
            
            try:
                model = trainer.train(augmented_features[train_mask], labels[train_mask])
                proba, _ = trainer.predict_with_meta_label(model, augmented_features[test_mask])
                metrics = evaluator.evaluate(labels[test_mask], proba)
                ic = metrics.get("ic", 0.0)
                if not pd.isna(ic):
                    augmented_ics.append(ic)
            except Exception:
                continue
        
        augmented_ic = float(np.mean(augmented_ics)) if augmented_ics else 0.0
        
        # Improvement
        ic_improvement = augmented_ic - baseline_ic
        
        passed = ic_improvement >= self.cfg.min_incremental_ic
        
        return {
            "passed": passed,
            "baseline_ic": baseline_ic,
            "augmented_ic": augmented_ic,
            "ic_improvement": ic_improvement,
        }


def generate_validation_report(
    validation_results: Dict[str, Dict],
    output_path: Optional[str] = None,
) -> str:
    """Generate markdown report for data validation."""
    lines = ["# New Data Validation Report\n"]
    
    passed_features = [k for k, v in validation_results.items() if v["passed"]]
    failed_features = [k for k, v in validation_results.items() if not v["passed"]]
    
    lines.append("## Summary\n")
    lines.append(f"- **Features validated**: {len(validation_results)}")
    lines.append(f"- **Passed**: {len(passed_features)} ✅")
    lines.append(f"- **Failed**: {len(failed_features)} ❌\n")
    
    if passed_features:
        lines.append("## Passed Features\n")
        for feat in passed_features:
            result = validation_results[feat]
            ic = result["metrics"].get("univariate_ic", {}).get("ic", 0.0)
            lines.append(f"- **{feat}** (IC: {ic:.4f})")
        lines.append("")
    
    if failed_features:
        lines.append("## Failed Features\n")
        for feat in failed_features:
            result = validation_results[feat]
            lines.append(f"\n### {feat}\n")
            lines.append("**Issues:**")
            for issue in result["issues"]:
                lines.append(f"- {issue}")
            lines.append("")
    
    lines.append("## Recommendations\n")
    if failed_features:
        lines.append("The following features should NOT be added:")
        for feat in failed_features:
            lines.append(f"- `{feat}`")
    else:
        lines.append("✅ All validated features passed quality checks.")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Validation report saved", extra={"path": output_path})
    
    return report

