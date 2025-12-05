"""
Alerting System for Model and Data Issues

Centralized alerting that:
1. Aggregates alerts from all monitoring systems
2. Prioritizes by severity
3. Generates actionable notifications
4. Logs alert history for analysis

Integrates with:
- Data quality checker
- Model decay detector
- Risk management (drawdown alerts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import json
import pandas as pd

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "INFO"           # Informational, no action needed
    WARNING = "WARNING"     # Attention needed, not urgent
    CRITICAL = "CRITICAL"   # Immediate attention required
    EMERGENCY = "EMERGENCY" # Stop trading immediately


class AlertCategory(Enum):
    """Alert categories."""
    
    DATA_QUALITY = "DATA_QUALITY"
    MODEL_PERFORMANCE = "MODEL_PERFORMANCE"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    SYSTEM = "SYSTEM"
    TRADING = "TRADING"


@dataclass
class Alert:
    """Single alert instance."""
    
    id: str
    timestamp: datetime
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    source: str  # Which system generated the alert
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "context": self.context,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class AlertConfig:
    """Alerting system configuration."""
    
    # Storage
    alert_log_path: Path = field(
        default_factory=lambda: settings.data_dir / "logs" / "alerts.jsonl"
    )
    
    # Thresholds for auto-generating alerts
    ic_warning_threshold: float = 0.02
    ic_critical_threshold: float = 0.0
    drawdown_warning_threshold: float = 0.10
    drawdown_critical_threshold: float = 0.15
    
    # Rate limiting
    max_alerts_per_hour: int = 50
    cooldown_minutes: int = 15  # Min time between same alert type


class AlertManager:
    """
    Central alert management system.
    
    Usage:
        manager = AlertManager()
        
        # Generate alerts from various sources
        manager.check_data_quality(quality_report)
        manager.check_model_health(decay_report)
        manager.check_drawdown(current_drawdown)
        
        # Get active alerts
        active = manager.get_active_alerts()
        for alert in active:
            print(f"[{alert.severity.value}] {alert.title}")
        
        # Acknowledge and resolve
        manager.acknowledge_alert(alert_id)
        manager.resolve_alert(alert_id, "Fixed by refreshing data")
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.alerts: List[Alert] = []
        self._alert_counter = 0
        self._last_alert_times: Dict[str, datetime] = {}
        
        # Ensure log directory exists
        self.config.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing alerts
        self._load_alerts()
    
    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ALT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"
    
    def _should_throttle(self, alert_type: str) -> bool:
        """Check if alert should be throttled."""
        last_time = self._last_alert_times.get(alert_type)
        if last_time is None:
            return False
        
        elapsed = (datetime.now() - last_time).total_seconds() / 60
        return elapsed < self.config.cooldown_minutes
    
    def create_alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Alert]:
        """
        Create and register a new alert.
        
        Returns:
            Alert if created, None if throttled
        """
        alert_type = f"{category.value}:{title}"
        
        if self._should_throttle(alert_type):
            logger.debug(f"Throttling alert: {title}")
            return None
        
        alert = Alert(
            id=self._generate_id(),
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            title=title,
            message=message,
            source=source,
            context=context or {},
        )
        
        self.alerts.append(alert)
        self._last_alert_times[alert_type] = datetime.now()
        self._log_alert(alert)
        
        # Log based on severity
        log_msg = f"[{severity.value}] {title}: {message}"
        if severity == AlertSeverity.EMERGENCY:
            logger.critical(log_msg)
        elif severity == AlertSeverity.CRITICAL:
            logger.error(log_msg)
        elif severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        return alert
    
    def check_data_quality(self, quality_report) -> List[Alert]:
        """
        Generate alerts from data quality report.
        
        Args:
            quality_report: DataQualityReport from data_quality module
        """
        alerts = []
        
        # Check overall level
        if hasattr(quality_report, 'overall_level'):
            level = quality_report.overall_level.value
            
            if level == "CRITICAL":
                alert = self.create_alert(
                    category=AlertCategory.DATA_QUALITY,
                    severity=AlertSeverity.CRITICAL,
                    title="Data Quality Critical",
                    message="Data quality check failed. Do not trade.",
                    source="DataQualityChecker",
                    context={"recommendations": quality_report.recommendations},
                )
                if alert:
                    alerts.append(alert)
            
            elif level == "STALE":
                alert = self.create_alert(
                    category=AlertCategory.DATA_QUALITY,
                    severity=AlertSeverity.WARNING,
                    title="Data is Stale",
                    message=f"Price data is {quality_report.price_freshness.get('days_stale', 'N/A')} days old",
                    source="DataQualityChecker",
                )
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def check_model_health(self, decay_report) -> List[Alert]:
        """
        Generate alerts from model decay report.
        
        Args:
            decay_report: DecayReport from decay_detector module
        """
        alerts = []
        
        if not hasattr(decay_report, 'alert_level'):
            return alerts
        
        level = decay_report.alert_level.value
        
        if level == "RED":
            alert = self.create_alert(
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.CRITICAL,
                title="Model Performance Degraded",
                message=f"IC: {decay_report.current_ic:.3f}, Precision: {decay_report.current_precision:.2%}",
                source="ModelDecayDetector",
                context={
                    "ic": decay_report.current_ic,
                    "precision": decay_report.current_precision,
                    "recommendations": decay_report.recommendations,
                },
            )
            if alert:
                alerts.append(alert)
        
        elif level == "YELLOW":
            alert = self.create_alert(
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                title="Model Performance Warning",
                message="Model metrics showing early signs of decay",
                source="ModelDecayDetector",
            )
            if alert:
                alerts.append(alert)
        
        if hasattr(decay_report, 'regime_shift_detected') and decay_report.regime_shift_detected:
            alert = self.create_alert(
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                title="Market Regime Shift Detected",
                message="Prediction distribution has changed significantly",
                source="ModelDecayDetector",
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def check_drawdown(
        self,
        current_drawdown: float,
        peak_equity: float,
        current_equity: float,
    ) -> Optional[Alert]:
        """
        Generate alerts based on portfolio drawdown.
        
        Args:
            current_drawdown: Current drawdown percentage (0.10 = 10%)
            peak_equity: Peak portfolio value
            current_equity: Current portfolio value
        """
        if current_drawdown >= self.config.drawdown_critical_threshold:
            return self.create_alert(
                category=AlertCategory.RISK_MANAGEMENT,
                severity=AlertSeverity.CRITICAL,
                title="Drawdown Limit Breached",
                message=f"Portfolio down {current_drawdown:.1%} from peak",
                source="RiskManager",
                context={
                    "drawdown": current_drawdown,
                    "peak": peak_equity,
                    "current": current_equity,
                },
            )
        
        elif current_drawdown >= self.config.drawdown_warning_threshold:
            return self.create_alert(
                category=AlertCategory.RISK_MANAGEMENT,
                severity=AlertSeverity.WARNING,
                title="Drawdown Warning",
                message=f"Portfolio down {current_drawdown:.1%} from peak",
                source="RiskManager",
                context={
                    "drawdown": current_drawdown,
                    "peak": peak_equity,
                    "current": current_equity,
                },
            )
        
        return None
    
    def check_ic(self, current_ic: float, rolling_window: int = 20) -> Optional[Alert]:
        """Check IC and generate alert if needed."""
        if current_ic < self.config.ic_critical_threshold:
            return self.create_alert(
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.CRITICAL,
                title="IC Below Critical Threshold",
                message=f"Rolling {rolling_window}d IC is {current_ic:.3f}",
                source="ICMonitor",
            )
        
        elif current_ic < self.config.ic_warning_threshold:
            return self.create_alert(
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                title="IC Below Warning Threshold",
                message=f"Rolling {rolling_window}d IC is {current_ic:.3f}",
                source="ICMonitor",
            )
        
        return None
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
    ) -> List[Alert]:
        """Get all unresolved alerts, optionally filtered."""
        active = [a for a in self.alerts if not a.resolved]
        
        if severity:
            active = [a for a in active if a.severity == severity]
        
        if category:
            active = [a for a in active if a.category == category]
        
        # Sort by severity (most severe first)
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }
        active.sort(key=lambda a: (severity_order.get(a.severity, 99), a.timestamp))
        
        return active
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self._log_alert(alert)
                return True
        return False
    
    def resolve_alert(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_notes = notes
                self._log_alert(alert)
                return True
        return False
    
    def _log_alert(self, alert: Alert) -> None:
        """Append alert to log file."""
        try:
            with open(self.config.alert_log_path, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def _load_alerts(self) -> None:
        """Load alerts from log file."""
        if not self.config.alert_log_path.exists():
            return
        
        try:
            alerts_by_id = {}
            with open(self.config.alert_log_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        alert = Alert(
                            id=data["id"],
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            category=AlertCategory(data["category"]),
                            severity=AlertSeverity(data["severity"]),
                            title=data["title"],
                            message=data["message"],
                            source=data["source"],
                            context=data.get("context", {}),
                            acknowledged=data.get("acknowledged", False),
                            resolved=data.get("resolved", False),
                            resolution_notes=data.get("resolution_notes"),
                        )
                        # Later entries override earlier ones (for updates)
                        alerts_by_id[alert.id] = alert
                    except Exception:
                        continue
            
            self.alerts = list(alerts_by_id.values())
            # Update counter
            if self.alerts:
                max_num = max(
                    int(a.id.split("-")[-1]) for a in self.alerts
                    if a.id.startswith("ALT-")
                )
                self._alert_counter = max_num
                
        except Exception as e:
            logger.warning(f"Could not load alert history: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current alert status."""
        active = self.get_active_alerts()
        
        by_severity = {}
        for sev in AlertSeverity:
            count = len([a for a in active if a.severity == sev])
            if count > 0:
                by_severity[sev.value] = count
        
        by_category = {}
        for cat in AlertCategory:
            count = len([a for a in active if a.category == cat])
            if count > 0:
                by_category[cat.value] = count
        
        return {
            "total_active": len(active),
            "by_severity": by_severity,
            "by_category": by_category,
            "most_severe": active[0].to_dict() if active else None,
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def send_alert(
    category: AlertCategory,
    severity: AlertSeverity,
    title: str,
    message: str,
    source: str = "manual",
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Alert]:
    """Convenience function to send an alert."""
    manager = get_alert_manager()
    return manager.create_alert(category, severity, title, message, source, context)
