"""IC Monitoring and Automatic Retraining System.

Monitors the Information Coefficient (IC) of the model on a rolling basis
and triggers automatic retraining when IC degrades below threshold.

Key Features:
- Daily IC tracking
- Rolling IC with configurable window
- Automatic retraining trigger
- Alert system for IC degradation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ICMetrics:
    """IC metrics for a specific period."""
    date: datetime
    ic: float
    rank_ic: float
    n_samples: int
    is_significant: bool  # p-value < 0.05


@dataclass
class ICMonitorConfig:
    """Configuration for IC monitoring."""
    ic_threshold: float = 0.02  # Trigger retraining if IC drops below
    rolling_window_days: int = 30  # Window for rolling IC
    min_samples_per_day: int = 50  # Minimum samples for valid IC
    alert_threshold: float = 0.03  # Alert if IC drops below (before retraining)
    check_frequency_days: int = 1  # How often to check IC


class ICMonitor:
    """Monitors IC and triggers retraining when necessary.
    
    Usage:
        monitor = ICMonitor()
        monitor.update(predictions_df, returns_df)
        
        if monitor.should_retrain():
            # Trigger retraining pipeline
            ...
    """
    
    def __init__(
        self,
        config: Optional[ICMonitorConfig] = None,
        storage_path: Optional[Path] = None
    ):
        self.config = config or ICMonitorConfig()
        self.storage_path = storage_path or Path(settings.experiments_dir) / "ic_monitor"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._metrics_file = self.storage_path / "ic_metrics.json"
        self._metrics_history: List[Dict] = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load historical IC metrics."""
        if self._metrics_file.exists():
            try:
                return json.loads(self._metrics_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load IC history: {e}")
        return []
    
    def _save_history(self) -> None:
        """Save IC metrics history."""
        self._metrics_file.write_text(json.dumps(self._metrics_history, indent=2, default=str))
    
    def compute_ic(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        method: str = 'spearman'
    ) -> Tuple[float, float]:
        """Compute IC between predictions and returns.
        
        Args:
            predictions: Model probability scores
            returns: Actual forward returns
            method: 'spearman' (rank IC) or 'pearson'
            
        Returns:
            (ic, p_value)
        """
        # Align and drop NaN
        valid = pd.DataFrame({'pred': predictions, 'ret': returns}).dropna()
        
        if len(valid) < self.config.min_samples_per_day:
            return np.nan, np.nan
        
        if method == 'spearman':
            ic, pval = spearmanr(valid['pred'], valid['ret'])
        else:
            ic = valid['pred'].corr(valid['ret'])
            pval = 0.0  # Approximate
        
        return float(ic), float(pval)
    
    def update(
        self,
        predictions_df: pd.DataFrame,
        date: Optional[datetime] = None
    ) -> ICMetrics:
        """Update IC metrics with new predictions.
        
        Args:
            predictions_df: DataFrame with columns:
                - date
                - ticker
                - proba (model prediction)
                - excess_fwd or label (actual return/outcome)
            date: Date to compute IC for (defaults to latest)
            
        Returns:
            ICMetrics for the computed period
        """
        df = predictions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        if date is None:
            date = df['date'].max()
        
        # Get data for the target date
        day_df = df[df['date'] == date]
        
        if len(day_df) < self.config.min_samples_per_day:
            logger.warning(f"Insufficient samples for IC computation: {len(day_df)}")
            return ICMetrics(
                date=date,
                ic=np.nan,
                rank_ic=np.nan,
                n_samples=len(day_df),
                is_significant=False
            )
        
        # Compute IC with forward returns
        returns_col = 'excess_fwd' if 'excess_fwd' in day_df.columns else 'label'
        
        ic, pval = self.compute_ic(day_df['proba'], day_df[returns_col], method='pearson')
        rank_ic, rank_pval = self.compute_ic(day_df['proba'], day_df[returns_col], method='spearman')
        
        metrics = ICMetrics(
            date=date,
            ic=ic,
            rank_ic=rank_ic,
            n_samples=len(day_df),
            is_significant=rank_pval < 0.05 if not np.isnan(rank_pval) else False
        )
        
        # Store in history
        self._metrics_history.append({
            'date': date.isoformat(),
            'ic': ic,
            'rank_ic': rank_ic,
            'n_samples': len(day_df),
            'is_significant': metrics.is_significant
        })
        self._save_history()
        
        return metrics
    
    def compute_rolling_ic(
        self,
        predictions_df: pd.DataFrame,
        window_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Compute rolling IC over a window.
        
        Args:
            predictions_df: Full predictions DataFrame
            window_days: Rolling window size (defaults to config)
            
        Returns:
            DataFrame with date and rolling IC
        """
        window = window_days or self.config.rolling_window_days
        df = predictions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        returns_col = 'excess_fwd' if 'excess_fwd' in df.columns else 'label'
        
        # Group by date and compute daily IC
        daily_ics = []
        for date, group in df.groupby('date'):
            if len(group) >= self.config.min_samples_per_day:
                ic, _ = self.compute_ic(group['proba'], group[returns_col])
                rank_ic, _ = self.compute_ic(group['proba'], group[returns_col], method='spearman')
                daily_ics.append({
                    'date': date,
                    'ic': ic,
                    'rank_ic': rank_ic,
                    'n_samples': len(group)
                })
        
        if not daily_ics:
            return pd.DataFrame()
        
        result = pd.DataFrame(daily_ics)
        result = result.sort_values('date')
        
        # Compute rolling mean
        result['rolling_ic'] = result['ic'].rolling(window, min_periods=5).mean()
        result['rolling_rank_ic'] = result['rank_ic'].rolling(window, min_periods=5).mean()
        
        return result
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained based on IC metrics.
        
        Returns:
            True if retraining is recommended
        """
        if not self._metrics_history:
            return False
        
        # Get recent metrics
        recent = self._metrics_history[-self.config.rolling_window_days:]
        
        if len(recent) < 5:
            return False
        
        # Compute rolling IC
        valid_ics = [m['rank_ic'] for m in recent if not np.isnan(m.get('rank_ic', np.nan))]
        
        if not valid_ics:
            return False
        
        rolling_ic = np.mean(valid_ics)
        
        if rolling_ic < self.config.ic_threshold:
            logger.warning(
                f"IC degradation detected: rolling IC = {rolling_ic:.4f} "
                f"(threshold: {self.config.ic_threshold})"
            )
            return True
        
        return False
    
    def get_alert_status(self) -> Dict:
        """Get current alert status.
        
        Returns:
            Dict with status, current IC, and recommendations
        """
        if not self._metrics_history:
            return {
                'status': 'no_data',
                'message': 'No IC history available',
                'rolling_ic': None,
                'recommendations': ['Run model predictions to start monitoring']
            }
        
        recent = self._metrics_history[-self.config.rolling_window_days:]
        valid_ics = [m['rank_ic'] for m in recent if not np.isnan(m.get('rank_ic', np.nan))]
        
        if not valid_ics:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough valid IC measurements',
                'rolling_ic': None,
                'recommendations': ['Ensure predictions have actual returns for IC computation']
            }
        
        rolling_ic = np.mean(valid_ics)
        
        if rolling_ic < self.config.ic_threshold:
            return {
                'status': 'critical',
                'message': f'IC below retraining threshold: {rolling_ic:.4f}',
                'rolling_ic': rolling_ic,
                'recommendations': [
                    'Trigger automatic retraining',
                    'Consider feature refresh',
                    'Review recent market regime'
                ]
            }
        elif rolling_ic < self.config.alert_threshold:
            return {
                'status': 'warning',
                'message': f'IC declining: {rolling_ic:.4f}',
                'rolling_ic': rolling_ic,
                'recommendations': [
                    'Monitor closely',
                    'Reduce position sizes',
                    'Investigate alpha decay'
                ]
            }
        else:
            return {
                'status': 'healthy',
                'message': f'IC stable: {rolling_ic:.4f}',
                'rolling_ic': rolling_ic,
                'recommendations': ['Continue normal operations']
            }
    
    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of IC history."""
        if not self._metrics_history:
            return {}
        
        valid_ics = [m['rank_ic'] for m in self._metrics_history 
                     if not np.isnan(m.get('rank_ic', np.nan))]
        
        if not valid_ics:
            return {}
        
        return {
            'total_days': len(self._metrics_history),
            'valid_days': len(valid_ics),
            'mean_ic': np.mean(valid_ics),
            'std_ic': np.std(valid_ics),
            'min_ic': np.min(valid_ics),
            'max_ic': np.max(valid_ics),
            'positive_ic_pct': np.mean([ic > 0 for ic in valid_ics]) * 100,
            'latest_ic': valid_ics[-1] if valid_ics else None,
            'first_date': self._metrics_history[0].get('date'),
            'last_date': self._metrics_history[-1].get('date'),
        }


class AutoRetrainer:
    """Automatic retraining orchestrator.
    
    Triggers retraining when IC drops below threshold.
    """
    
    def __init__(
        self,
        experiment_name: str,
        config_path: Optional[Path] = None
    ):
        self.experiment_name = experiment_name
        self.config_path = config_path
        self.monitor = ICMonitor()
    
    def check_and_retrain(
        self,
        predictions_df: pd.DataFrame,
        force: bool = False
    ) -> bool:
        """Check IC and retrain if necessary.
        
        Args:
            predictions_df: Recent predictions with actual returns
            force: Force retraining regardless of IC
            
        Returns:
            True if retraining was triggered
        """
        # Update IC metrics
        self.monitor.update(predictions_df)
        
        # Check if retraining is needed
        if force or self.monitor.should_retrain():
            logger.info("Triggering automatic retraining...")
            return self._trigger_retraining()
        
        return False
    
    def _trigger_retraining(self) -> bool:
        """Trigger the retraining pipeline."""
        import subprocess
        
        cmd = [
            "python", "scripts/run_training.py",
            "--experiment", self.experiment_name,
        ]
        
        if self.config_path:
            cmd.extend(["--config", str(self.config_path)])
        
        try:
            logger.info(f"Running retraining: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=Path(settings.experiments_dir).parent,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Retraining completed successfully")
                return True
            else:
                logger.error(f"Retraining failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
            return False


def get_ic_monitor(storage_path: Optional[Path] = None) -> ICMonitor:
    """Factory function to get IC monitor."""
    return ICMonitor(storage_path=storage_path)


def get_auto_retrainer(experiment_name: str) -> AutoRetrainer:
    """Factory function to get auto retrainer."""
    return AutoRetrainer(experiment_name)
