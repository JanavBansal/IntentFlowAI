"""Walk-Forward Optimization (WFO) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from intentflow_ai.config.settings import LightGBMConfig
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardTrainer:
    """Orchestrates rolling/expanding window training."""

    lgbm_cfg: LightGBMConfig
    initial_train_days: int = 730  # 2 years
    step_days: int = 30  # 1 month
    embargo_days: int = 0

    def train_and_predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "label",
        date_col: str = "date",
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Run WFO and return stitched predictions and feature importance history.

        Args:
            df: DataFrame containing features, target, and date.
            feature_cols: List of feature column names.
            target_col: Name of target column.
            date_col: Name of date column.

        Returns:
            Tuple of:
            - pd.Series: Out-of-sample predictions (probabilities), indexed by df.index.
            - pd.DataFrame: Feature importance history (one row per step).
        """
        df = df.sort_values(date_col).copy()
        dates = df[date_col].unique()
        dates = np.sort(dates)

        if len(dates) == 0:
            raise ValueError("No dates in dataframe.")

        # Find start index
        start_date = dates[0]
        # We need at least initial_train_days of history
        # Find the first date that is >= start_date + initial_train_days
        cutoff_time = pd.Timestamp(start_date) + pd.Timedelta(days=self.initial_train_days)
        
        # Find the index in 'dates' that corresponds to the first split point
        split_indices = [i for i, d in enumerate(dates) if pd.Timestamp(d) >= cutoff_time]
        
        if not split_indices:
            raise ValueError(
                f"Dataset span ({df[date_col].max() - df[date_col].min()}) "
                f"is shorter than initial_train_days ({self.initial_train_days} days)."
            )
        
        start_idx = split_indices[0]
        
        # Generate split points (indices in the unique dates array)
        # We step by approx step_days (converted to number of trading days? No, use calendar days)
        # Better: Iterate by date
        
        predictions = pd.Series(index=df.index, dtype=float)
        importances = []
        
        current_date = pd.Timestamp(dates[start_idx])
        end_date = pd.Timestamp(dates[-1])
        
        trainer = LightGBMTrainer(self.lgbm_cfg)
        
        # Loop until we cover the whole dataset
        pbar = tqdm(total=(end_date - current_date).days // self.step_days + 1, desc="WFO Steps")
        
        while current_date <= end_date:
            # Define Train and Test masks
            # Train: [Start, Current_Date - Embargo]
            # Test: (Current_Date, Current_Date + Step]
            
            train_end = current_date - pd.Timedelta(days=self.embargo_days)
            test_end = current_date + pd.Timedelta(days=self.step_days)
            
            train_mask = df[date_col] <= train_end
            test_mask = (df[date_col] > current_date) & (df[date_col] <= test_end)
            
            if not train_mask.any():
                logger.warning(f"No training data for split {current_date}")
                current_date = test_end
                pbar.update(1)
                continue
                
            if not test_mask.any():
                # No test data in this window (maybe gap in trading), just advance
                current_date = test_end
                pbar.update(1)
                continue
            
            # Train
            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, target_col]
            
            # Simple check for single class
            if y_train.nunique() < 2:
                logger.warning(f"Single class in training target for split {current_date}. Skipping.")
                current_date = test_end
                pbar.update(1)
                continue
                
            model = trainer.train(X_train, y_train)
            
            # Predict
            X_test = df.loc[test_mask, feature_cols]
            proba, _ = trainer.predict_with_meta_label(model, X_test)
            
            # Store
            predictions.loc[test_mask] = proba.values
            
            # Feature Importance
            imp = trainer.feature_importance(model)
            imp["date"] = current_date
            importances.append(imp)
            
            # Advance
            current_date = test_end
            pbar.update(1)
            
        pbar.close()
        
        importance_df = pd.DataFrame(importances)
        if not importance_df.empty:
            importance_df = importance_df.set_index("date")
            
        return predictions.dropna(), importance_df
