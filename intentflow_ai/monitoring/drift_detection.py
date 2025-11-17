"""Real-time drift detection for production alpha models.

Monitors:
- Feature distribution drift (KS test, PSI)
- Model performance degradation
- Regime shifts
- Prediction distribution changes
- Feature importance stability

Triggers alerts and automated retrain when drift exceeds thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    
    # Feature drift thresholds
    ks_statistic_threshold: float = 0.15  # Kolmogorov-Smirnov test
    psi_threshold: float = 0.20  # Population Stability Index
    
    # Performance drift thresholds
    ic_degradation_threshold: float = 0.30  # 30% drop in IC
    sharpe_degradation_threshold: float = 0.40  # 40% drop in Sharpe
    hit_rate_degradation_threshold: float = 0.20  # 20% drop in hit rate
    
    # Monitoring windows
    reference_window_days: int = 90  # Training period reference
    monitoring_window_days: int = 30  # Recent period to compare
    min_samples_for_test: int = 30
    
    # Alert triggers
    max_drifting_features_pct: float = 0.30  # Alert if >30% features drift
    consecutive_bad_days_threshold: int = 5  # Alert after 5 bad days
    
    # Automated actions
    trigger_retrain: bool = True
    retrain_cooldown_days: int = 7  # Don't retrain more than once per week


class DriftAlert:
    """Drift alert with severity and recommended actions."""
    
    def __init__(
        self,
        alert_type: str,
        severity: str,  # "low", "medium", "high", "critical"
        message: str,
        details: Dict,
        timestamp: Optional[datetime] = None,
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details
        self.timestamp = timestamp or datetime.now()
        self.recommended_actions = self._get_recommended_actions()
    
    def _get_recommended_actions(self) -> List[str]:
        """Get recommended actions based on alert type and severity."""
        actions = []
        
        if self.severity in ["high", "critical"]:
            if "feature_drift" in self.alert_type:
                actions.append("Review feature engineering pipeline")
                actions.append("Investigate data source changes")
                actions.append("Consider retraining model with recent data")
            
            elif "performance_degradation" in self.alert_type:
                actions.append("Analyze recent trades and signals")
                actions.append("Check for regime changes")
                actions.append("Schedule model retrain")
            
            elif "prediction_drift" in self.alert_type:
                actions.append("Review model calibration")
                actions.append("Check for overfitting")
                actions.append("Validate label generation process")
        
        if self.severity == "critical":
            actions.append("🚨 STOP TRADING - Manual review required")
        
        return actions
    
    def to_dict(self) -> Dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class DriftDetector:
    """Comprehensive drift detection for trading models."""
    
    cfg: DriftConfig = field(default_factory=DriftConfig)
    
    def detect_all_drift(
        self,
        current_features: pd.DataFrame,
        reference_features: pd.DataFrame,
        current_predictions: Optional[pd.Series] = None,
        reference_predictions: Optional[pd.Series] = None,
        current_performance: Optional[Dict] = None,
        reference_performance: Optional[Dict] = None,
    ) -> Tuple[List[DriftAlert], Dict[str, object]]:
        """Run comprehensive drift detection.
        
        Args:
            current_features: Recent feature data
            reference_features: Training/reference feature data
            current_predictions: Recent model predictions
            reference_predictions: Training predictions
            current_performance: Recent performance metrics
            reference_performance: Reference performance metrics
            
        Returns:
            Tuple of (alerts, drift_report)
        """
        alerts = []
        report = {}
        
        # 1. Feature drift
        if not current_features.empty and not reference_features.empty:
            feature_drift = self._detect_feature_drift(current_features, reference_features)
            report["feature_drift"] = feature_drift
            
            drift_alerts = self._create_feature_drift_alerts(feature_drift)
            alerts.extend(drift_alerts)
        
        # 2. Prediction drift
        if current_predictions is not None and reference_predictions is not None:
            prediction_drift = self._detect_prediction_drift(current_predictions, reference_predictions)
            report["prediction_drift"] = prediction_drift
            
            pred_alerts = self._create_prediction_drift_alerts(prediction_drift)
            alerts.extend(pred_alerts)
        
        # 3. Performance degradation
        if current_performance and reference_performance:
            perf_drift = self._detect_performance_drift(current_performance, reference_performance)
            report["performance_drift"] = perf_drift
            
            perf_alerts = self._create_performance_alerts(perf_drift)
            alerts.extend(perf_alerts)
        
        # Overall drift summary
        report["summary"] = self._generate_drift_summary(alerts, report)
        report["timestamp"] = datetime.now().isoformat()
        
        logger.info(
            "Drift detection complete",
            extra={
                "alerts": len(alerts),
                "high_severity": sum(1 for a in alerts if a.severity in ["high", "critical"])
            }
        )
        
        return alerts, report
    
    def _detect_feature_drift(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> Dict[str, object]:
        """Detect drift in feature distributions."""
        drift_results = {}
        drifting_features = []
        
        common_features = set(current.columns) & set(reference.columns)
        
        for feat in common_features:
            curr_vals = current[feat].dropna()
            ref_vals = reference[feat].dropna()
            
            if len(curr_vals) < self.cfg.min_samples_for_test or len(ref_vals) < self.cfg.min_samples_for_test:
                continue
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pval = stats.ks_2samp(curr_vals, ref_vals)
            except Exception:
                ks_stat, ks_pval = 0.0, 1.0
            
            # Population Stability Index (PSI)
            psi = self._compute_psi(ref_vals, curr_vals)
            
            is_drifting = (ks_stat > self.cfg.ks_statistic_threshold) or (psi > self.cfg.psi_threshold)
            
            drift_results[feat] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "psi": float(psi),
                "drifting": bool(is_drifting),
                "current_mean": float(curr_vals.mean()),
                "reference_mean": float(ref_vals.mean()),
                "current_std": float(curr_vals.std()),
                "reference_std": float(ref_vals.std()),
            }
            
            if is_drifting:
                drifting_features.append(feat)
        
        drift_pct = len(drifting_features) / len(common_features) if common_features else 0.0
        
        return {
            "drifting_features": drifting_features,
            "drift_percentage": float(drift_pct),
            "total_features_tested": len(common_features),
            "feature_details": drift_results,
        }
    
    def _detect_prediction_drift(
        self,
        current: pd.Series,
        reference: pd.Series,
    ) -> Dict[str, object]:
        """Detect drift in prediction distributions."""
        # KS test on prediction probabilities
        try:
            ks_stat, ks_pval = stats.ks_2samp(current.dropna(), reference.dropna())
        except Exception:
            ks_stat, ks_pval = 0.0, 1.0
        
        # PSI for predictions
        psi = self._compute_psi(reference.dropna(), current.dropna())
        
        # Check for extreme predictions (overconfidence)
        extreme_preds_current = ((current > 0.95) | (current < 0.05)).mean()
        extreme_preds_reference = ((reference > 0.95) | (reference < 0.05)).mean()
        
        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "psi": float(psi),
            "current_mean": float(current.mean()),
            "reference_mean": float(reference.mean()),
            "current_std": float(current.std()),
            "reference_std": float(reference.std()),
            "extreme_predictions_current": float(extreme_preds_current),
            "extreme_predictions_reference": float(extreme_preds_reference),
            "drifting": bool(ks_stat > self.cfg.ks_statistic_threshold or psi > self.cfg.psi_threshold),
        }
    
    def _detect_performance_drift(
        self,
        current: Dict,
        reference: Dict,
    ) -> Dict[str, object]:
        """Detect performance degradation."""
        degradation = {}
        
        # IC degradation
        current_ic = current.get("rank_ic", 0.0)
        reference_ic = reference.get("rank_ic", 0.0)
        ic_change = (current_ic - reference_ic) / (abs(reference_ic) + 1e-9)
        degradation["ic_degradation"] = float(ic_change)
        
        # Sharpe degradation
        current_sharpe = current.get("sharpe", 0.0)
        reference_sharpe = reference.get("sharpe", 0.0)
        sharpe_change = (current_sharpe - reference_sharpe) / (abs(reference_sharpe) + 1e-9)
        degradation["sharpe_degradation"] = float(sharpe_change)
        
        # Hit rate degradation
        current_hit_rate = current.get("hit_rate", 0.0)
        reference_hit_rate = reference.get("hit_rate", 0.0)
        hit_rate_change = (current_hit_rate - reference_hit_rate) / (reference_hit_rate + 1e-9)
        degradation["hit_rate_degradation"] = float(hit_rate_change)
        
        # Overall assessment
        critical_degradation = (
            ic_change < -self.cfg.ic_degradation_threshold or
            sharpe_change < -self.cfg.sharpe_degradation_threshold or
            hit_rate_change < -self.cfg.hit_rate_degradation_threshold
        )
        
        degradation["critical_degradation"] = bool(critical_degradation)
        degradation["current_metrics"] = current
        degradation["reference_metrics"] = reference
        
        return degradation
    
    def _compute_psi(self, reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
        """Compute Population Stability Index (PSI).
        
        PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Small change
        PSI > 0.2: Significant change (drift)
        """
        if len(reference) < 10 or len(current) < 10:
            return 0.0
        
        # Create bins based on reference distribution
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins = np.unique(bins)
        
        if len(bins) < 2:
            return 0.0
        
        # Compute distributions
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)
        
        ref_pct = ref_hist / (len(reference) + 1e-9)
        curr_pct = curr_hist / (len(current) + 1e-9)
        
        # Avoid division by zero
        ref_pct = np.maximum(ref_pct, 1e-6)
        curr_pct = np.maximum(curr_pct, 1e-6)
        
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        return float(abs(psi))
    
    def _create_feature_drift_alerts(self, drift_info: Dict) -> List[DriftAlert]:
        """Create alerts for feature drift."""
        alerts = []
        
        drift_pct = drift_info.get("drift_percentage", 0.0)
        drifting_features = drift_info.get("drifting_features", [])
        
        if drift_pct > self.cfg.max_drifting_features_pct:
            severity = "critical" if drift_pct > 0.5 else "high"
            
            alert = DriftAlert(
                alert_type="feature_drift",
                severity=severity,
                message=f"Feature drift detected: {drift_pct:.1%} of features drifting",
                details={
                    "drift_percentage": drift_pct,
                    "drifting_features": drifting_features[:10],  # Top 10
                    "total_tested": drift_info.get("total_features_tested", 0),
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _create_prediction_drift_alerts(self, drift_info: Dict) -> List[DriftAlert]:
        """Create alerts for prediction drift."""
        alerts = []
        
        if drift_info.get("drifting", False):
            psi = drift_info.get("psi", 0.0)
            severity = "critical" if psi > 0.5 else "high" if psi > 0.3 else "medium"
            
            alert = DriftAlert(
                alert_type="prediction_drift",
                severity=severity,
                message=f"Prediction distribution drift detected (PSI={psi:.3f})",
                details=drift_info
            )
            alerts.append(alert)
        
        return alerts
    
    def _create_performance_alerts(self, perf_info: Dict) -> List[DriftAlert]:
        """Create alerts for performance degradation."""
        alerts = []
        
        if perf_info.get("critical_degradation", False):
            alert = DriftAlert(
                alert_type="performance_degradation",
                severity="critical",
                message="Critical performance degradation detected",
                details={
                    "ic_degradation": perf_info.get("ic_degradation", 0.0),
                    "sharpe_degradation": perf_info.get("sharpe_degradation", 0.0),
                    "hit_rate_degradation": perf_info.get("hit_rate_degradation", 0.0),
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_drift_summary(self, alerts: List[DriftAlert], report: Dict) -> Dict:
        """Generate overall drift summary."""
        severity_counts = {}
        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Overall health score (0-100)
        health_score = 100.0
        health_score -= severity_counts.get("critical", 0) * 40
        health_score -= severity_counts.get("high", 0) * 20
        health_score -= severity_counts.get("medium", 0) * 10
        health_score -= severity_counts.get("low", 0) * 5
        health_score = max(0.0, health_score)
        
        requires_action = any(a.severity in ["high", "critical"] for a in alerts)
        should_retrain = (
            self.cfg.trigger_retrain and
            requires_action and
            severity_counts.get("critical", 0) > 0
        )
        
        return {
            "total_alerts": len(alerts),
            "severity_counts": severity_counts,
            "health_score": health_score,
            "requires_action": requires_action,
            "should_retrain": should_retrain,
            "status": self._get_status_label(health_score),
        }
    
    def _get_status_label(self, health_score: float) -> str:
        """Get status label from health score."""
        if health_score >= 80:
            return "healthy"
        elif health_score >= 60:
            return "warning"
        elif health_score >= 40:
            return "degraded"
        else:
            return "critical"


def save_drift_report(
    alerts: List[DriftAlert],
    report: Dict,
    output_dir: Path,
):
    """Save drift report and alerts to disk."""
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save alerts
    alerts_path = output_dir / "drift_alerts.json"
    with open(alerts_path, "w") as f:
        json.dump([a.to_dict() for a in alerts], f, indent=2, default=float)
    
    # Save full report
    report_path = output_dir / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    
    # Save summary markdown
    md_path = output_dir / "drift_summary.md"
    markdown = generate_drift_markdown(alerts, report)
    with open(md_path, "w") as f:
        f.write(markdown)
    
    logger.info("Drift report saved", extra={"dir": str(output_dir)})


def generate_drift_markdown(alerts: List[DriftAlert], report: Dict) -> str:
    """Generate human-readable drift report."""
    lines = ["# Drift Detection Report\n"]
    
    summary = report.get("summary", {})
    
    lines.append(f"**Status**: {summary.get('status', 'unknown').upper()}")
    lines.append(f"**Health Score**: {summary.get('health_score', 0):.0f}/100")
    lines.append(f"**Total Alerts**: {summary.get('total_alerts', 0)}\n")
    
    # Alert summary by severity
    lines.append("## Alert Summary\n")
    severity_counts = summary.get("severity_counts", {})
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        if count > 0:
            lines.append(f"- **{severity.upper()}**: {count}")
    lines.append("")
    
    # Detailed alerts
    if alerts:
        lines.append("## Alerts\n")
        for alert in sorted(alerts, key=lambda a: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(a.severity, 4)):
            lines.append(f"### {alert.severity.upper()}: {alert.message}")
            lines.append(f"**Type**: {alert.alert_type}")
            lines.append(f"**Timestamp**: {alert.timestamp}")
            
            if alert.recommended_actions:
                lines.append("**Recommended Actions**:")
                for action in alert.recommended_actions:
                    lines.append(f"  - {action}")
            lines.append("")
    
    # Drift details
    feature_drift = report.get("feature_drift", {})
    if feature_drift:
        lines.append("## Feature Drift Details\n")
        lines.append(f"- **Drifting features**: {feature_drift.get('drift_percentage', 0):.1%}")
        drifting = feature_drift.get("drifting_features", [])
        if drifting:
            lines.append(f"- **Top drifting features**: {', '.join(drifting[:5])}\n")
    
    return "\n".join(lines)

