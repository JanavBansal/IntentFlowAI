"""Ensemble methods for robust out-of-sample performance.

This module implements diverse ensemble strategies:
1. Parameter diversity (different regularization strengths)
2. Feature diversity (different feature subsets)
3. Temporal diversity (different training windows)
4. Algorithmic diversity (LightGBM, XGBoost, CatBoost, Ridge)
5. Bagging with purged bootstrap
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from intentflow_ai.config.settings import LightGBMConfig
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.modeling.evaluation import ModelEvaluator
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Multi-Algorithm Ensemble (NEW)
# ============================================================================

class BaseModelTrainer(ABC):
    """Abstract base class for model trainers."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass


class LGBMModelTrainer(BaseModelTrainer):
    """LightGBM trainer wrapper."""
    
    def __init__(self, config: Optional[LightGBMConfig] = None):
        self.config = config or LightGBMConfig()
        self._trainer = LightGBMTrainer(self.config)
    
    @property
    def name(self) -> str:
        return "lightgbm"
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        return self._trainer.train(X, y)
    
    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        proba, _ = self._trainer.predict_with_meta_label(model, X)
        return proba.values


class XGBModelTrainer(BaseModelTrainer):
    """XGBoost trainer."""
    
    def __init__(self, **kwargs):
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 5,
            "learning_rate": 0.02,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        self.params.update(kwargs)
    
    @property
    def name(self) -> str:
        return "xgboost"
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        model = xgb.XGBClassifier(**self.params)
        model.fit(X, y, verbose=False)
        return model
    
    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)[:, 1]


class CatBoostModelTrainer(BaseModelTrainer):
    """CatBoost trainer."""
    
    def __init__(self, **kwargs):
        self.params = {
            "iterations": 500,
            "learning_rate": 0.02,
            "depth": 5,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        self.params.update(kwargs)
    
    @property
    def name(self) -> str:
        return "catboost"
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        model = CatBoostClassifier(**self.params)
        model.fit(X, y)
        return model
    
    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)[:, 1]


class RidgeModelTrainer(BaseModelTrainer):
    """Ridge logistic regression trainer (linear baseline)."""
    
    def __init__(self, **kwargs):
        self.params = {
            "C": 1.0,  # Inverse regularization
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        }
        self.params.update(kwargs)
    
    @property
    def name(self) -> str:
        return "ridge"
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Ridge needs scaling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**self.params)),
        ])
        pipeline.fit(X, y)
        return pipeline
    
    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)[:, 1]


@dataclass
class MultiAlgoEnsembleConfig:
    """Configuration for multi-algorithm ensemble."""
    
    # Model weights (must sum to 1.0)
    # These are FIXED weights, not optimized (to prevent overfitting)
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "lightgbm": 0.35,
        "xgboost": 0.30,
        "catboost": 0.20,
        "ridge": 0.15,
    })
    
    # Which models to include
    include_lightgbm: bool = True
    include_xgboost: bool = True
    include_catboost: bool = True
    include_ridge: bool = True
    
    # Aggregation method
    aggregation: str = "weighted"  # "weighted", "mean", "median"


class MultiAlgoEnsemble:
    """
    Multi-algorithm ensemble combining different model types.
    
    Uses FIXED weights (not optimized) to prevent overfitting.
    
    Usage:
        ensemble = MultiAlgoEnsemble()
        ensemble.train(X_train, y_train)
        proba = ensemble.predict(X_test)
    """
    
    def __init__(self, config: Optional[MultiAlgoEnsembleConfig] = None):
        self.config = config or MultiAlgoEnsembleConfig()
        self.trainers: Dict[str, BaseModelTrainer] = {}
        self.models: Dict[str, Any] = {}
        self._setup_trainers()
    
    def _setup_trainers(self):
        """Initialize model trainers based on config."""
        if self.config.include_lightgbm:
            self.trainers["lightgbm"] = LGBMModelTrainer()
        
        if self.config.include_xgboost:
            try:
                self.trainers["xgboost"] = XGBModelTrainer()
            except Exception as e:
                logger.warning(f"XGBoost not available: {e}")
        
        if self.config.include_catboost:
            try:
                self.trainers["catboost"] = CatBoostModelTrainer()
            except Exception as e:
                logger.warning(f"CatBoost not available: {e}")
        
        if self.config.include_ridge:
            self.trainers["ridge"] = RidgeModelTrainer()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> "MultiAlgoEnsemble":
        """
        Train all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Self (for chaining)
        """
        logger.info(
            "Training multi-algo ensemble",
            extra={"models": list(self.trainers.keys()), "n_samples": len(X)}
        )
        
        for name, trainer in self.trainers.items():
            try:
                logger.info(f"Training {name}...")
                model = trainer.train(X, y)
                self.models[name] = model
                logger.info(f"  {name} trained successfully")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        if not self.models:
            raise ValueError("All models failed to train!")
        
        logger.info(
            "Multi-algo ensemble training complete",
            extra={"models_trained": list(self.models.keys())}
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        if not self.models:
            raise ValueError("No trained models. Call train() first.")
        
        predictions = {}
        weights = {}
        
        for name, model in self.models.items():
            trainer = self.trainers[name]
            proba = trainer.predict_proba(model, X)
            predictions[name] = proba
            weights[name] = self.config.model_weights.get(name, 1.0 / len(self.models))
        
        # Normalize weights for available models
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate
        if self.config.aggregation == "weighted":
            ensemble_proba = np.zeros(len(X))
            for name, proba in predictions.items():
                ensemble_proba += proba * weights[name]
        elif self.config.aggregation == "median":
            probas_array = np.column_stack(list(predictions.values()))
            ensemble_proba = np.median(probas_array, axis=1)
        else:  # mean
            probas_array = np.column_stack(list(predictions.values()))
            ensemble_proba = np.mean(probas_array, axis=1)
        
        return ensemble_proba
    
    def predict_with_details(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Generate predictions with individual model outputs.
        
        Returns:
            Tuple of (ensemble_proba, details_dict)
        """
        if not self.models:
            raise ValueError("No trained models. Call train() first.")
        
        predictions = {}
        
        for name, model in self.models.items():
            trainer = self.trainers[name]
            predictions[name] = trainer.predict_proba(model, X)
        
        ensemble_proba = self.predict(X)
        
        # Compute uncertainty (std across models)
        probas_array = np.column_stack(list(predictions.values()))
        uncertainty = np.std(probas_array, axis=1)
        
        details = {
            "individual_predictions": predictions,
            "uncertainty": uncertainty,
            "models_used": list(self.models.keys()),
            "weights": {k: self.config.model_weights.get(k, 0) for k in self.models.keys()},
        }
        
        return ensemble_proba, details
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get aggregated feature importance across models.
        
        Returns:
            DataFrame with feature importances per model and average
        """
        importances = {}
        
        for name, model in self.models.items():
            if name == "lightgbm":
                imp = model.feature_importance(importance_type="gain")
                importances[name] = imp
            elif name == "xgboost":
                imp = model.feature_importances_
                importances[name] = imp
            elif name == "catboost":
                imp = model.feature_importances_
                importances[name] = imp
            # Ridge doesn't have feature importances in the same way
        
        if not importances:
            return pd.DataFrame()
        
        # Normalize and average
        df = pd.DataFrame(importances)
        
        # Normalize each column to sum to 1
        for col in df.columns:
            total = df[col].sum()
            if total > 0:
                df[col] = df[col] / total
        
        df["average"] = df.mean(axis=1)
        df = df.sort_values("average", ascending=False)
        
        return df


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""
    
    # Ensemble strategy
    n_models: int = 5  # Number of diverse models
    
    # Diversity sources
    use_parameter_diversity: bool = True
    use_feature_diversity: bool = True
    use_temporal_diversity: bool = False
    
    # Parameter ranges for diversity
    learning_rate_range: Tuple[float, float] = (0.01, 0.05)
    reg_lambda_range: Tuple[float, float] = (0.5, 5.0)
    subsample_range: Tuple[float, float] = (0.7, 0.9)
    feature_fraction_range: Tuple[float, float] = (0.6, 0.8)
    
    # Feature diversity
    feature_subsample_pct: float = 0.8  # Use 80% of features per model
    
    # Temporal diversity (use different date ranges)
    temporal_window_pct: float = 0.8  # Use 80% of training data per model
    
    # Ensemble aggregation
    aggregation: str = "mean"  # "mean", "median", "weighted"
    
    # Pruning low-quality models
    prune_low_performers: bool = True
    min_ic_threshold: float = 0.01  # Drop models with IC < 1%


@dataclass
class DiverseEnsemble:
    """Train and manage ensemble of diverse models."""
    
    cfg: EnsembleConfig = field(default_factory=EnsembleConfig)
    models: List = field(default_factory=list)
    model_configs: List[LightGBMConfig] = field(default_factory=list)
    model_features: List[List[str]] = field(default_factory=list)
    model_weights: List[float] = field(default_factory=list)
    model_performance: List[Dict] = field(default_factory=list)
    
    def train_ensemble(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        *,
        base_config: LightGBMConfig,
        validation_features: Optional[pd.DataFrame] = None,
        validation_labels: Optional[pd.Series] = None,
    ) -> DiverseEnsemble:
        """Train ensemble of diverse models.
        
        Args:
            features: Training features
            labels: Training labels
            dates: Training dates
            base_config: Base LightGBM configuration
            validation_features: Optional validation set for quality assessment
            validation_labels: Optional validation labels
            
        Returns:
            Self (for chaining)
        """
        logger.info(
            "Training diverse ensemble",
            extra={"n_models": self.cfg.n_models, "n_features": len(features.columns)}
        )
        
        rng = np.random.default_rng(42)
        
        for i in range(self.cfg.n_models):
            logger.info(f"Training ensemble model {i+1}/{self.cfg.n_models}")
            
            # Generate diverse configuration
            model_cfg = self._generate_diverse_config(base_config, i, rng)
            
            # Select diverse feature subset
            feature_subset = self._select_feature_subset(
                list(features.columns), i, rng
            )
            
            # Select temporal subset (if enabled)
            train_mask = self._select_temporal_subset(dates, i, rng)
            
            # Train model
            trainer = LightGBMTrainer(model_cfg)
            
            train_features = features.loc[train_mask, feature_subset]
            train_labels = labels[train_mask]
            
            try:
                model = trainer.train(train_features, train_labels)
                
                # Evaluate on validation set
                performance = {}
                if validation_features is not None and validation_labels is not None:
                    val_features_subset = validation_features[feature_subset]
                    proba, _ = trainer.predict_with_meta_label(model, val_features_subset)
                    
                    evaluator = ModelEvaluator(horizon_days=10)
                    metrics = evaluator.evaluate(validation_labels, proba)
                    
                    performance = {
                        "ic": float(metrics.get("ic", 0.0)),
                        "rank_ic": float(metrics.get("rank_ic", 0.0)),
                        "roc_auc": float(metrics.get("roc_auc", 0.5)),
                    }
                    
                    logger.info(
                        f"Model {i+1} validation performance",
                        extra=performance
                    )
                
                # Store model
                self.models.append(model)
                self.model_configs.append(model_cfg)
                self.model_features.append(feature_subset)
                self.model_performance.append(performance)
                
            except Exception as exc:
                logger.warning(f"Failed to train model {i+1}: {exc}")
                continue
        
        # Prune low-performing models if enabled
        if self.cfg.prune_low_performers and validation_features is not None:
            self._prune_models()
        
        # Compute weights
        self._compute_weights()
        
        logger.info(
            "Ensemble training complete",
            extra={"models_trained": len(self.models)}
        )
        
        return self
    
    def predict(
        self,
        features: pd.DataFrame,
    ) -> Tuple[pd.Series, Dict]:
        """Generate ensemble predictions.
        
        Args:
            features: Feature matrix
            
        Returns:
            Tuple of (ensemble_probabilities, metadata_dict)
        """
        if not self.models:
            raise ValueError("No models in ensemble. Call train_ensemble first.")
        
        all_probas = []
        
        for i, (model, feature_subset) in enumerate(zip(self.models, self.model_features)):
            # Get predictions from this model
            features_subset = features[feature_subset]
            trainer = LightGBMTrainer(self.model_configs[i])
            proba, _ = trainer.predict_with_meta_label(model, features_subset)
            
            all_probas.append(proba.values)
        
        # Aggregate predictions
        probas_array = np.column_stack(all_probas)
        
        if self.cfg.aggregation == "mean":
            ensemble_proba = np.mean(probas_array, axis=1)
        elif self.cfg.aggregation == "median":
            ensemble_proba = np.median(probas_array, axis=1)
        elif self.cfg.aggregation == "weighted":
            # Weighted average by model weights
            if not self.model_weights:
                ensemble_proba = np.mean(probas_array, axis=1)
            else:
                weights = np.array(self.model_weights[:len(all_probas)])
                weights = weights / weights.sum()  # Normalize
                ensemble_proba = np.average(probas_array, axis=1, weights=weights)
        else:
            ensemble_proba = np.mean(probas_array, axis=1)
        
        # Compute prediction uncertainty (standard deviation across models)
        uncertainty = np.std(probas_array, axis=1)
        
        metadata = {
            "n_models": len(self.models),
            "aggregation": self.cfg.aggregation,
            "mean_uncertainty": float(np.mean(uncertainty)),
            "individual_predictions": probas_array.tolist(),
            "uncertainty": uncertainty.tolist(),
        }
        
        return pd.Series(ensemble_proba, index=features.index), metadata
    
    def _generate_diverse_config(
        self,
        base_config: LightGBMConfig,
        model_idx: int,
        rng: np.random.Generator,
    ) -> LightGBMConfig:
        """Generate configuration with parameter diversity."""
        if not self.cfg.use_parameter_diversity:
            return base_config
        
        # Create copy of base config
        from dataclasses import replace
        
        cfg = base_config
        
        # Vary learning rate
        lr_min, lr_max = self.cfg.learning_rate_range
        learning_rate = float(rng.uniform(lr_min, lr_max))
        
        # Vary regularization
        reg_min, reg_max = self.cfg.reg_lambda_range
        reg_lambda = float(rng.uniform(reg_min, reg_max))
        
        # Vary subsampling
        sub_min, sub_max = self.cfg.subsample_range
        subsample = float(rng.uniform(sub_min, sub_max))
        
        # Vary feature fraction
        ff_min, ff_max = self.cfg.feature_fraction_range
        feature_fraction = float(rng.uniform(ff_min, ff_max))
        
        # Create new config with varied parameters
        cfg = replace(
            cfg,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            subsample=subsample,
            feature_fraction=feature_fraction,
            random_state=base_config.random_state + model_idx,  # Different seed
        )
        
        return cfg
    
    def _select_feature_subset(
        self,
        all_features: List[str],
        model_idx: int,
        rng: np.random.Generator,
    ) -> List[str]:
        """Select diverse feature subset for this model."""
        if not self.cfg.use_feature_diversity:
            return all_features
        
        # Sample features randomly
        n_features = max(
            int(len(all_features) * self.cfg.feature_subsample_pct),
            min(10, len(all_features))  # At least 10 features
        )
        
        selected = rng.choice(all_features, size=n_features, replace=False).tolist()
        
        return selected
    
    def _select_temporal_subset(
        self,
        dates: pd.Series,
        model_idx: int,
        rng: np.random.Generator,
    ) -> pd.Series:
        """Select temporal subset for this model (if temporal diversity enabled)."""
        if not self.cfg.use_temporal_diversity:
            return pd.Series(True, index=dates.index)
        
        # Sample date range
        unique_dates = sorted(dates.unique())
        n_dates = max(
            int(len(unique_dates) * self.cfg.temporal_window_pct),
            min(100, len(unique_dates))
        )
        
        # Random contiguous window
        start_idx = rng.integers(0, len(unique_dates) - n_dates + 1)
        selected_dates = unique_dates[start_idx:start_idx + n_dates]
        
        mask = dates.isin(selected_dates)
        
        return mask
    
    def _prune_models(self):
        """Remove low-performing models from ensemble."""
        if not self.model_performance:
            return
        
        # Filter by IC threshold
        valid_indices = []
        for i, perf in enumerate(self.model_performance):
            ic = perf.get("ic", 0.0)
            if ic >= self.cfg.min_ic_threshold:
                valid_indices.append(i)
        
        if not valid_indices:
            logger.warning("All models below IC threshold, keeping all")
            return
        
        # Keep only valid models
        self.models = [self.models[i] for i in valid_indices]
        self.model_configs = [self.model_configs[i] for i in valid_indices]
        self.model_features = [self.model_features[i] for i in valid_indices]
        self.model_performance = [self.model_performance[i] for i in valid_indices]
        
        logger.info(
            "Pruned low-performing models",
            extra={"kept": len(valid_indices), "removed": self.cfg.n_models - len(valid_indices)}
        )
    
    def _compute_weights(self):
        """Compute model weights based on validation performance."""
        if not self.model_performance:
            # Equal weights
            self.model_weights = [1.0 / len(self.models)] * len(self.models)
            return
        
        # Weight by IC (higher IC = higher weight)
        ics = [perf.get("ic", 0.0) for perf in self.model_performance]
        
        # Ensure all positive
        min_ic = min(ics)
        if min_ic < 0:
            ics = [ic - min_ic + 0.01 for ic in ics]
        
        # Normalize to sum to 1
        total = sum(ics)
        if total > 0:
            self.model_weights = [ic / total for ic in ics]
        else:
            self.model_weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info(
            "Computed model weights",
            extra={"weights": [f"{w:.3f}" for w in self.model_weights]}
        )


def evaluate_ensemble_diversity(
    ensemble: DiverseEnsemble,
    features: pd.DataFrame,
) -> Dict[str, float]:
    """Measure ensemble diversity metrics.
    
    High diversity = more robust ensemble.
    """
    if not ensemble.models or len(ensemble.models) < 2:
        return {"diversity": 0.0}
    
    # Get predictions from all models
    all_probas = []
    for i, (model, feature_subset) in enumerate(zip(ensemble.models, ensemble.model_features)):
        features_subset = features[feature_subset]
        trainer = LightGBMTrainer(ensemble.model_configs[i])
        proba, _ = trainer.predict_with_meta_label(model, features_subset)
        all_probas.append(proba.values)
    
    probas_array = np.column_stack(all_probas)
    
    # Diversity metric 1: Average pairwise correlation (lower = more diverse)
    n_models = len(all_probas)
    pairwise_corrs = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr = np.corrcoef(probas_array[:, i], probas_array[:, j])[0, 1]
            if not np.isnan(corr):
                pairwise_corrs.append(corr)
    
    avg_correlation = float(np.mean(pairwise_corrs)) if pairwise_corrs else 1.0
    
    # Diversity metric 2: Prediction disagreement (higher = more diverse)
    prediction_std = np.std(probas_array, axis=1).mean()
    
    # Diversity metric 3: Q-statistic (measure of diversity)
    # Lower Q = more diverse
    q_statistics = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Binary predictions at 0.5 threshold
            pred_i = (probas_array[:, i] > 0.5).astype(int)
            pred_j = (probas_array[:, j] > 0.5).astype(int)
            
            # Q-statistic calculation
            n11 = np.sum((pred_i == 1) & (pred_j == 1))
            n10 = np.sum((pred_i == 1) & (pred_j == 0))
            n01 = np.sum((pred_i == 0) & (pred_j == 1))
            n00 = np.sum((pred_i == 0) & (pred_j == 0))
            
            denom = (n11 * n00 + n10 * n01)
            if denom > 0:
                q = (n11 * n00 - n10 * n01) / denom
                q_statistics.append(q)
    
    avg_q_stat = float(np.mean(q_statistics)) if q_statistics else 1.0
    
    # Overall diversity score (0-1, higher = more diverse)
    diversity_score = (
        (1 - avg_correlation) * 0.4 +  # Low correlation = diverse
        prediction_std * 0.3 +  # High std = diverse
        (1 - abs(avg_q_stat)) * 0.3  # Low Q = diverse
    )
    
    return {
        "diversity_score": float(diversity_score),
        "avg_correlation": float(avg_correlation),
        "prediction_std": float(prediction_std),
        "avg_q_statistic": float(avg_q_stat),
        "n_models": n_models,
    }


def generate_ensemble_report(
    ensemble: DiverseEnsemble,
    diversity_metrics: Dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate ensemble summary report."""
    lines = ["# Ensemble Model Report\n"]
    
    lines.append("## Configuration\n")
    lines.append(f"- **Number of models**: {len(ensemble.models)}")
    lines.append(f"- **Aggregation method**: {ensemble.cfg.aggregation}")
    lines.append(f"- **Parameter diversity**: {ensemble.cfg.use_parameter_diversity}")
    lines.append(f"- **Feature diversity**: {ensemble.cfg.use_feature_diversity}")
    lines.append(f"- **Temporal diversity**: {ensemble.cfg.use_temporal_diversity}\n")
    
    lines.append("## Diversity Metrics\n")
    lines.append(f"- **Diversity score**: {diversity_metrics.get('diversity_score', 0):.3f}")
    lines.append(f"- **Avg correlation**: {diversity_metrics.get('avg_correlation', 0):.3f}")
    lines.append(f"- **Prediction std**: {diversity_metrics.get('prediction_std', 0):.3f}")
    lines.append(f"- **Q-statistic**: {diversity_metrics.get('avg_q_statistic', 0):.3f}\n")
    
    if ensemble.model_performance:
        lines.append("## Individual Model Performance\n")
        lines.append("| Model | IC | Rank IC | ROC AUC | Weight |")
        lines.append("|-------|----|---------|---------| ------ |")
        
        for i, perf in enumerate(ensemble.model_performance):
            weight = ensemble.model_weights[i] if i < len(ensemble.model_weights) else 0.0
            lines.append(
                f"| Model {i+1} | "
                f"{perf.get('ic', 0):.4f} | "
                f"{perf.get('rank_ic', 0):.4f} | "
                f"{perf.get('roc_auc', 0):.4f} | "
                f"{weight:.3f} |"
            )
        lines.append("")
    
    lines.append("## Recommendation\n")
    diversity = diversity_metrics.get("diversity_score", 0)
    if diversity > 0.3:
        lines.append("✅ **Good diversity** - Ensemble is well-diversified")
    elif diversity > 0.15:
        lines.append("⚠️ **Moderate diversity** - Consider increasing diversity")
    else:
        lines.append("❌ **Low diversity** - Models are too similar")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Ensemble report saved", extra={"path": output_path})
    
    return report

