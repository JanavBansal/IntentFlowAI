"""
Model Decay Detector

Monitors model performance over time to detect degradation:
- Rolling IC calculation
- Rolling Precision@K
- Regime shift detection
- Alert system with severity levels

Critical for knowing when to retrain the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    
    GREEN = "GREEN"    # All metrics healthy
    YELLOW = "YELLOW"  # Some degradation, monitor closely
    RED = "RED"        # Significant degradation, retrain recommended


@dataclass
class DecayConfig:
    """Configuration for decay detection."""
    
    # Rolling window for metrics calculation
    rolling_window_days: int = 30
    
    # Minimum observations for valid calculation
    min_observations: int = 50
    
    # IC thresholds
    ic_green_threshold: float = 0.03  # Healthy IC
    ic_yellow_threshold: float = 0.02  # Warning
    ic_red_threshold: float = 0.01  # Critical
    
    # Precision@10 thresholds (as decimal)
    precision_green_threshold: float = 0.60  # 60%
    precision_yellow_threshold: float = 0.50  # 50%
    precision_red_threshold: float = 0.40  # 40%
    
    # Regime shift detection
    regime_shift_zscore: float = 2.0  # Z-score for shift detection
    regime_lookback_days: int = 60
    
    # Alert cooldown (don't spam alerts)
    alert_cooldown_hours: int = 24


@dataclass
class DecayReport:
    """Model decay monitoring report."""
    
    timestamp: datetime
    alert_level: AlertLevel
    
    # Core metrics
    rolling_ic_30d: float
    rolling_precision_30d: float
    rolling_hit_rate_30d: float
    
    # Trend
    ic_trend: str  # "improving", "stable", "declining"
    precision_trend: str
    
    # Regime
    regime_shift_detected: bool
    regime_details: Optional[str]
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "alert_level": self.alert_level.value,
            "rolling_ic_30d": self.rolling_ic_30d,
            "rolling_precision_30d": self.rolling_precision_30d,
            "rolling_hit_rate_30d": self.rolling_hit_rate_30d,
            "ic_trend": self.ic_trend,
            "precision_trend": self.precision_trend,
            "regime_shift_detected": self.regime_shift_detected,
            "regime_details": self.regime_details,
            "recommendations": self.recommendations,
        }
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"MODEL HEALTH REPORT - {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"STATUS: {self.alert_level.value}",
            "=" * 60,
            "",
            "METRICS (30-day rolling):",
            f"  Information Coefficient: {self.rolling_ic_30d:.4f} ({self.ic_trend})",
            f"  Precision@10: {self.rolling_precision_30d*100:.1f}% ({self.precision_trend})",
            f"  Hit Rate: {self.rolling_hit_rate_30d*100:.1f}%",
            "",
        ]
        
        if self.regime_shift_detected:
            lines.extend([
                "⚠️ REGIME SHIFT DETECTED",
                f"  Details: {self.regime_details}",
                "",
            ])
        
        if self.recommendations:
            lines.append("RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)


class ModelDecayDetector:
    """
    Monitor model performance and detect degradation.
    
    Usage:
        detector = ModelDecayDetector()
        
        # Check model health
        report = detector.check_health(predictions_df)
        print(report.to_summary())
        
        # Get alert level
        if report.alert_level == AlertLevel.RED:
            trigger_retrain()
    """
    
    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self._last_alert_time: Optional[datetime] = None
        self._historical_metrics: List[Dict] = []
    
    def check_health(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: Optional[pd.DataFrame] = None,
    ) -> DecayReport:
        """
        Check model health and generate report.
        
        Args:
            predictions_df: DataFrame with columns [date, ticker, proba, label]
                           where label is the actual outcome (realized)
            actuals_df: Optional separate DataFrame with actuals
            
        Returns:
            DecayReport with health metrics and recommendations
        """
        df = self._prepare_data(predictions_df, actuals_df)
        
        # Calculate rolling metrics
        ic_30d = self._calculate_rolling_ic(df)
        precision_30d = self._calculate_rolling_precision(df)
        hit_rate_30d = self._calculate_rolling_hit_rate(df)
        
        # Calculate trends
        ic_trend = self._detect_trend(df, "ic")
        precision_trend = self._detect_trend(df, "precision")
        
        # Detect regime shift
        regime_shift, regime_details = self._detect_regime_shift(df)
        
        # Determine alert level
        alert_level = self._determine_alert_level(
            ic_30d, precision_30d, regime_shift
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            alert_level, ic_30d, precision_30d, ic_trend, precision_trend, regime_shift
        )
        
        # Store for history
        self._historical_metrics.append({
            "timestamp": datetime.now(),
            "ic": ic_30d,
            "precision": precision_30d,
            "alert_level": alert_level.value,
        })
        
        report = DecayReport(
            timestamp=datetime.now(),
            alert_level=alert_level,
            rolling_ic_30d=ic_30d,
            rolling_precision_30d=precision_30d,
            rolling_hit_rate_30d=hit_rate_30d,
            ic_trend=ic_trend,
            precision_trend=precision_trend,
            regime_shift_detected=regime_shift,
            regime_details=regime_details,
            recommendations=recommendations,
        )
        
        # Log report
        log_extra = {
            "alert_level": alert_level.value,
            "ic_30d": f"{ic_30d:.4f}",
            "precision_30d": f"{precision_30d:.2%}",
        }
        
        if alert_level == AlertLevel.RED:
            logger.error("Model health check: CRITICAL", extra=log_extra)
        elif alert_level == AlertLevel.YELLOW:
            logger.warning("Model health check: WARNING", extra=log_extra)
        else:
            logger.info("Model health check: OK", extra=log_extra)
        
        return report
    
    def _prepare_data(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Prepare data for analysis."""
        df = predictions_df.copy()
        
        if actuals_df is not None:
            # Merge actuals
            df = df.merge(
                actuals_df[["date", "ticker", "label"]],
                on=["date", "ticker"],
                how="left",
                suffixes=("", "_actual"),
            )
            if "label_actual" in df.columns:
                df["label"] = df["label_actual"]
        
        # Ensure date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        return df
    
    def _calculate_rolling_ic(self, df: pd.DataFrame) -> float:
        """Calculate 30-day rolling Information Coefficient."""
        window_days = self.config.rolling_window_days
        
        if "date" not in df.columns or "proba" not in df.columns or "label" not in df.columns:
            return 0.0
        
        # Get recent data
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=window_days)
        recent = df[df["date"] >= cutoff]
        
        if len(recent) < self.config.min_observations:
            return np.nan
        
        # Remove NaN labels
        valid = recent.dropna(subset=["proba", "label"])
        
        if len(valid) < self.config.min_observations:
            return np.nan
        
        # Spearman correlation
        try:
            ic, _ = stats.spearmanr(valid["proba"], valid["label"])
            return float(ic) if not np.isnan(ic) else 0.0
        except Exception:
            return 0.0
    
    def _calculate_rolling_precision(self, df: pd.DataFrame, k: int = 10) -> float:
        """Calculate rolling Precision@K."""
        window_days = self.config.rolling_window_days
        
        if "date" not in df.columns:
            return 0.0
        
        # Get recent data
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=window_days)
        recent = df[df["date"] >= cutoff]
        
        # Group by date and calculate precision
        precisions = []
        
        for date, group in recent.groupby("date"):
            group = group.dropna(subset=["proba", "label"])
            if len(group) < k:
                continue
            
            # Top K by probability
            top_k = group.nlargest(k, "proba")
            
            # Precision = proportion of top K that are positive
            precision = top_k["label"].mean()
            precisions.append(precision)
        
        if not precisions:
            return np.nan
        
        return float(np.mean(precisions))
    
    def _calculate_rolling_hit_rate(self, df: pd.DataFrame) -> float:
        """Calculate rolling hit rate (accuracy at 0.5 threshold)."""
        window_days = self.config.rolling_window_days
        
        if "date" not in df.columns:
            return 0.5
        
        # Get recent data
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=window_days)
        recent = df[df["date"] >= cutoff].dropna(subset=["proba", "label"])
        
        if len(recent) < self.config.min_observations:
            return np.nan
        
        # Hit rate = correct predictions / total
        predictions = (recent["proba"] > 0.5).astype(int)
        actuals = recent["label"].astype(int)
        
        hit_rate = (predictions == actuals).mean()
        return float(hit_rate)
    
    def _detect_trend(self, df: pd.DataFrame, metric: str) -> str:
        """Detect trend in metric (improving/stable/declining)."""
        window_days = self.config.rolling_window_days
        
        if "date" not in df.columns:
            return "unknown"
        
        # Split into two periods
        max_date = df["date"].max()
        mid_date = max_date - pd.Timedelta(days=window_days // 2)
        cutoff = max_date - pd.Timedelta(days=window_days)
        
        first_half = df[(df["date"] >= cutoff) & (df["date"] < mid_date)]
        second_half = df[df["date"] >= mid_date]
        
        if len(first_half) < 20 or len(second_half) < 20:
            return "unknown"
        
        # Calculate metric for each half
        if metric == "ic":
            first_val = self._calculate_ic_single(first_half)
            second_val = self._calculate_ic_single(second_half)
        elif metric == "precision":
            first_val = self._calculate_precision_single(first_half)
            second_val = self._calculate_precision_single(second_half)
        else:
            return "unknown"
        
        if np.isnan(first_val) or np.isnan(second_val):
            return "unknown"
        
        # Determine trend
        change = second_val - first_val
        threshold = 0.01  # 1% change threshold
        
        if change > threshold:
            return "improving"
        elif change < -threshold:
            return "declining"
        else:
            return "stable"
    
    def _calculate_ic_single(self, df: pd.DataFrame) -> float:
        """Calculate IC for a single period."""
        valid = df.dropna(subset=["proba", "label"])
        if len(valid) < 20:
            return np.nan
        try:
            ic, _ = stats.spearmanr(valid["proba"], valid["label"])
            return float(ic)
        except Exception:
            return np.nan
    
    def _calculate_precision_single(self, df: pd.DataFrame, k: int = 10) -> float:
        """Calculate precision for a single period."""
        precisions = []
        for date, group in df.groupby("date"):
            group = group.dropna(subset=["proba", "label"])
            if len(group) < k:
                continue
            top_k = group.nlargest(k, "proba")
            precisions.append(top_k["label"].mean())
        return float(np.mean(precisions)) if precisions else np.nan
    
    def _detect_regime_shift(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Detect if market regime has shifted."""
        if "date" not in df.columns:
            return False, None
        
        lookback = self.config.regime_lookback_days
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=lookback)
        
        recent = df[df["date"] >= cutoff]
        
        if len(recent) < 30:
            return False, None
        
        # Check for distribution shift in predictions
        daily_proba_mean = recent.groupby("date")["proba"].mean()
        
        if len(daily_proba_mean) < 20:
            return False, None
        
        # Split into two halves and compare
        mid_idx = len(daily_proba_mean) // 2
        first_half = daily_proba_mean.iloc[:mid_idx]
        second_half = daily_proba_mean.iloc[mid_idx:]
        
        # Two-sample t-test
        try:
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            
            if abs(t_stat) > self.config.regime_shift_zscore and p_value < 0.05:
                direction = "higher" if t_stat > 0 else "lower"
                return True, f"Prediction distribution shifted {direction} (p={p_value:.3f})"
        except Exception:
            pass
        
        return False, None
    
    def _determine_alert_level(
        self,
        ic: float,
        precision: float,
        regime_shift: bool,
    ) -> AlertLevel:
        """Determine alert level based on metrics."""
        cfg = self.config
        
        # Check for RED conditions
        if (not np.isnan(ic) and ic < cfg.ic_red_threshold) or \
           (not np.isnan(precision) and precision < cfg.precision_red_threshold):
            return AlertLevel.RED
        
        # Check for YELLOW conditions
        if (not np.isnan(ic) and ic < cfg.ic_yellow_threshold) or \
           (not np.isnan(precision) and precision < cfg.precision_yellow_threshold) or \
           regime_shift:
            return AlertLevel.YELLOW
        
        return AlertLevel.GREEN
    
    def _generate_recommendations(
        self,
        alert_level: AlertLevel,
        ic: float,
        precision: float,
        ic_trend: str,
        precision_trend: str,
        regime_shift: bool,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if alert_level == AlertLevel.RED:
            recommendations.append("URGENT: Schedule model retraining within 1 week")
            recommendations.append("Review recent feature distributions for data quality issues")
            recommendations.append("Consider reducing position sizes until model is updated")
        
        elif alert_level == AlertLevel.YELLOW:
            if regime_shift:
                recommendations.append("Market regime shift detected - monitor closely")
                recommendations.append("Consider reviewing macro features and adjusting model")
            
            if ic_trend == "declining":
                recommendations.append("IC trend is declining - prepare for retraining")
            
            if precision_trend == "declining":
                recommendations.append("Precision declining - review top signal quality")
        
        else:  # GREEN
            if ic_trend == "declining":
                recommendations.append("Monitor: IC showing slight decline")
        
        if not recommendations:
            recommendations.append("Model performing within normal parameters")
            recommendations.append("Continue monitoring daily")
        
        return recommendations


def get_decay_detector() -> ModelDecayDetector:
    """Get configured decay detector."""
    return ModelDecayDetector()
