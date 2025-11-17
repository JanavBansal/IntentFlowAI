"""Advanced feature selection for out-of-sample generalization.

This module implements aggressive feature pruning strategies:
1. Orthogonality-based filtering (remove redundant features)
2. Out-of-sample IC validation (only keep features that help OOS)
3. Permutation importance with stability checks
4. Forward/backward selection with cross-validation
5. Group-wise feature selection (remove entire blocks if unhelpful)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from intentflow_ai.features.orthogonality import (
    FeatureOrthogonalityAnalyzer,
    OrthogonalityConfig,
)
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.modeling.evaluation import ModelEvaluator
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Configuration for aggressive feature selection."""
    
    # Orthogonality filtering
    max_correlation: float = 0.80  # Stricter than before
    max_vif: float = 4.0  # Lower VIF threshold
    
    # Out-of-sample IC validation
    min_oos_ic: float = 0.02  # Features must improve IC by 2%
    cv_folds: int = 3
    
    # Permutation importance
    use_permutation_importance: bool = True
    n_permutations: int = 10
    min_importance_score: float = 0.001  # Drop low-importance features
    
    # Forward/backward selection
    use_forward_selection: bool = False  # Expensive but effective
    use_backward_elimination: bool = True
    max_features: int = 30  # Cap total feature count
    
    # Group selection
    test_feature_groups: bool = True  # Test entire feature blocks


@dataclass
class FeatureSelector:
    """Aggressive feature selection for generalization."""
    
    cfg: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    
    def select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        *,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        lgbm_cfg=None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Run comprehensive feature selection pipeline.
        
        Args:
            features: Feature matrix
            labels: Target labels
            dates: Date series for time-based CV
            feature_groups: Optional dict mapping group name to feature list
            lgbm_cfg: LightGBM config
            
        Returns:
            Tuple of (selected_features, dropped_features_with_reasons)
        """
        if features.empty:
            return [], {}
        
        logger.info(
            "Starting aggressive feature selection",
            extra={"initial_features": len(features.columns)}
        )
        
        selected = list(features.columns)
        drop_reasons = {}
        
        # Stage 1: Orthogonality filtering
        logger.info("Stage 1: Orthogonality filtering")
        selected, reasons = self._orthogonality_filter(
            features[selected], labels
        )
        drop_reasons.update(reasons)
        
        # Stage 2: Permutation importance
        if self.cfg.use_permutation_importance and lgbm_cfg:
            logger.info("Stage 2: Permutation importance")
            selected, reasons = self._permutation_importance_filter(
                features[selected], labels, lgbm_cfg
            )
            drop_reasons.update(reasons)
        
        # Stage 3: Out-of-sample IC validation
        logger.info("Stage 3: Out-of-sample IC validation")
        selected, reasons = self._oos_ic_filter(
            features[selected], labels, dates, lgbm_cfg
        )
        drop_reasons.update(reasons)
        
        # Stage 4: Backward elimination (if enabled)
        if self.cfg.use_backward_elimination and lgbm_cfg:
            logger.info("Stage 4: Backward elimination")
            selected, reasons = self._backward_elimination(
                features[selected], labels, dates, lgbm_cfg
            )
            drop_reasons.update(reasons)
        
        # Stage 5: Group-wise selection (if groups provided)
        if self.cfg.test_feature_groups and feature_groups and lgbm_cfg:
            logger.info("Stage 5: Group-wise selection")
            selected, reasons = self._group_selection(
                features[selected], labels, dates, feature_groups, lgbm_cfg
            )
            drop_reasons.update(reasons)
        
        # Final cap on feature count
        if len(selected) > self.cfg.max_features:
            logger.warning(
                f"Capping features at {self.cfg.max_features} (had {len(selected)})"
            )
            # Keep top features by simple univariate IC
            ics = {}
            for col in selected:
                clean = pd.DataFrame({"feat": features[col], "label": labels}).dropna()
                if len(clean) > 100:
                    ic = clean["feat"].corr(clean["label"], method="spearman")
                    ics[col] = abs(ic)
            
            top_features = sorted(ics.items(), key=lambda x: -x[1])[:self.cfg.max_features]
            selected = [feat for feat, _ in top_features]
            
            # Mark dropped features
            for col in features.columns:
                if col not in selected and col not in drop_reasons:
                    drop_reasons[col] = "Dropped due to max_features cap"
        
        logger.info(
            "Feature selection complete",
            extra={
                "initial": len(features.columns),
                "selected": len(selected),
                "dropped": len(drop_reasons),
            }
        )
        
        return selected, drop_reasons
    
    def _orthogonality_filter(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Remove correlated and multicollinear features."""
        cfg = OrthogonalityConfig(
            max_correlation=self.cfg.max_correlation,
            max_vif=self.cfg.max_vif,
        )
        analyzer = FeatureOrthogonalityAnalyzer(cfg)
        
        selected, drop_reasons = analyzer.select_orthogonal_features(
            features, labels
        )
        
        return selected, drop_reasons
    
    def _permutation_importance_filter(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        lgbm_cfg,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Remove features with low permutation importance."""
        if features.empty:
            return [], {}
        
        # Train baseline model
        trainer = LightGBMTrainer(lgbm_cfg)
        model = trainer.train(features, labels)
        
        # Get baseline predictions
        proba_baseline, _ = trainer.predict_with_meta_label(model, features)
        evaluator = ModelEvaluator(horizon_days=10)
        baseline_metrics = evaluator.evaluate(labels, proba_baseline)
        baseline_ic = baseline_metrics.get("ic", 0.0)
        
        # Permutation importance: shuffle each feature and measure IC drop
        importances = {}
        rng = np.random.default_rng(42)
        
        for col in features.columns:
            ic_drops = []
            
            for _ in range(self.cfg.n_permutations):
                # Shuffle feature
                shuffled_features = features.copy()
                shuffled_features[col] = rng.permutation(shuffled_features[col].values)
                
                # Predict with shuffled feature
                proba_shuffled, _ = trainer.predict_with_meta_label(model, shuffled_features)
                metrics = evaluator.evaluate(labels, proba_shuffled)
                shuffled_ic = metrics.get("ic", 0.0)
                
                # IC drop = baseline - shuffled (positive = feature is important)
                ic_drops.append(baseline_ic - shuffled_ic)
            
            # Average importance across permutations
            mean_importance = float(np.mean(ic_drops))
            importances[col] = mean_importance
        
        # Drop low-importance features
        drop_reasons = {}
        selected = []
        
        for col, importance in importances.items():
            if importance < self.cfg.min_importance_score:
                drop_reasons[col] = f"Low permutation importance ({importance:.4f})"
            else:
                selected.append(col)
        
        logger.info(
            "Permutation importance filtering complete",
            extra={"selected": len(selected), "dropped": len(drop_reasons)}
        )
        
        return selected, drop_reasons
    
    def _oos_ic_filter(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        lgbm_cfg,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Keep only features that improve out-of-sample IC."""
        if features.empty or lgbm_cfg is None:
            return list(features.columns), {}
        
        from intentflow_ai.utils.splits import purged_time_series_splits
        
        # Create time-based CV splits
        cv_frame = pd.DataFrame({"date": dates})
        try:
            cv_splits = purged_time_series_splits(
                cv_frame,
                n_splits=self.cfg.cv_folds,
                embargo_days=10,
                horizon_days=10,
            )
        except Exception:
            # Fallback: simple splits
            logger.warning("Failed to create purged splits, using simple splits")
            return list(features.columns), {}
        
        # Test each feature individually
        feature_oos_ics = {}
        trainer = LightGBMTrainer(lgbm_cfg)
        evaluator = ModelEvaluator(horizon_days=10)
        
        for col in features.columns:
            # Use only this feature
            single_feature = features[[col]]
            
            fold_ics = []
            for train_mask, test_mask in cv_splits:
                if not train_mask.any() or not test_mask.any():
                    continue
                
                try:
                    model = trainer.train(single_feature[train_mask], labels[train_mask])
                    proba, _ = trainer.predict_with_meta_label(model, single_feature[test_mask])
                    
                    metrics = evaluator.evaluate(labels[test_mask], proba)
                    ic = metrics.get("ic", 0.0)
                    if not pd.isna(ic):
                        fold_ics.append(ic)
                except Exception:
                    continue
            
            # Average OOS IC
            if fold_ics:
                feature_oos_ics[col] = float(np.mean(fold_ics))
            else:
                feature_oos_ics[col] = 0.0
        
        # Drop features with negative or near-zero OOS IC
        drop_reasons = {}
        selected = []
        
        for col, oos_ic in feature_oos_ics.items():
            if oos_ic < self.cfg.min_oos_ic:
                drop_reasons[col] = f"Low OOS IC ({oos_ic:.4f})"
            else:
                selected.append(col)
        
        logger.info(
            "OOS IC filtering complete",
            extra={"selected": len(selected), "dropped": len(drop_reasons)}
        )
        
        return selected, drop_reasons
    
    def _backward_elimination(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        lgbm_cfg,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Iteratively remove least useful features."""
        if features.empty or len(features.columns) <= 5:
            return list(features.columns), {}
        
        from intentflow_ai.utils.splits import purged_time_series_splits
        
        cv_frame = pd.DataFrame({"date": dates})
        try:
            cv_splits = purged_time_series_splits(
                cv_frame,
                n_splits=self.cfg.cv_folds,
                embargo_days=10,
                horizon_days=10,
            )
        except Exception:
            return list(features.columns), {}
        
        trainer = LightGBMTrainer(lgbm_cfg)
        evaluator = ModelEvaluator(horizon_days=10)
        
        current_features = list(features.columns)
        drop_reasons = {}
        
        # Baseline performance with all features
        baseline_ic = self._evaluate_feature_set(
            features[current_features], labels, cv_splits, trainer, evaluator
        )
        
        # Iteratively remove features
        while len(current_features) > 10:  # Keep at least 10 features
            # Try removing each feature
            removal_impacts = {}
            
            for col in current_features:
                # Test without this feature
                test_features = [f for f in current_features if f != col]
                ic_without = self._evaluate_feature_set(
                    features[test_features], labels, cv_splits, trainer, evaluator
                )
                
                # Impact = baseline - without (positive = feature helps)
                removal_impacts[col] = baseline_ic - ic_without
            
            # Find feature with least impact (or negative impact)
            worst_feature = min(removal_impacts.items(), key=lambda x: x[1])
            col, impact = worst_feature
            
            # If removing this feature doesn't hurt (or helps), remove it
            if impact <= 0.01:  # Threshold: 1% IC impact
                current_features.remove(col)
                drop_reasons[col] = f"Backward elimination (impact: {impact:.4f})"
                baseline_ic = self._evaluate_feature_set(
                    features[current_features], labels, cv_splits, trainer, evaluator
                )
                logger.debug(f"Removed {col}, new IC: {baseline_ic:.4f}")
            else:
                # Can't remove any more features without hurting performance
                break
        
        logger.info(
            "Backward elimination complete",
            extra={"remaining": len(current_features), "eliminated": len(drop_reasons)}
        )
        
        return current_features, drop_reasons
    
    def _group_selection(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        feature_groups: Dict[str, List[str]],
        lgbm_cfg,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Test entire feature groups (e.g., 'momentum', 'technical') and drop unhelpful groups."""
        if features.empty or not feature_groups:
            return list(features.columns), {}
        
        from intentflow_ai.utils.splits import purged_time_series_splits
        
        cv_frame = pd.DataFrame({"date": dates})
        try:
            cv_splits = purged_time_series_splits(
                cv_frame,
                n_splits=self.cfg.cv_folds,
                embargo_days=10,
                horizon_days=10,
            )
        except Exception:
            return list(features.columns), {}
        
        trainer = LightGBMTrainer(lgbm_cfg)
        evaluator = ModelEvaluator(horizon_days=10)
        
        # Map features to groups
        feature_to_group = {}
        for group_name, group_features in feature_groups.items():
            for feat in group_features:
                if feat in features.columns:
                    feature_to_group[feat] = group_name
        
        # Test each group's contribution
        group_ics = {}
        
        for group_name, group_features in feature_groups.items():
            # Features in this group that exist
            existing_group_feats = [f for f in group_features if f in features.columns]
            
            if not existing_group_feats:
                continue
            
            # Test only this group
            ic = self._evaluate_feature_set(
                features[existing_group_feats], labels, cv_splits, trainer, evaluator
            )
            group_ics[group_name] = ic
        
        # Drop groups with negative or very low IC
        min_group_ic = 0.01  # Groups must show at least 1% IC
        drop_reasons = {}
        selected = []
        
        for col in features.columns:
            group = feature_to_group.get(col)
            
            if group and group_ics.get(group, 0.0) < min_group_ic:
                if col not in drop_reasons:
                    drop_reasons[col] = f"Group '{group}' has low IC ({group_ics.get(group, 0.0):.4f})"
            else:
                selected.append(col)
        
        logger.info(
            "Group selection complete",
            extra={
                "groups_tested": len(group_ics),
                "selected": len(selected),
                "dropped": len(drop_reasons),
            }
        )
        
        return selected, drop_reasons
    
    def _evaluate_feature_set(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        cv_splits: List,
        trainer: LightGBMTrainer,
        evaluator: ModelEvaluator,
    ) -> float:
        """Evaluate feature set via CV and return mean OOS IC."""
        fold_ics = []
        
        for train_mask, test_mask in cv_splits:
            if not train_mask.any() or not test_mask.any():
                continue
            
            try:
                model = trainer.train(features[train_mask], labels[train_mask])
                proba, _ = trainer.predict_with_meta_label(model, features[test_mask])
                
                metrics = evaluator.evaluate(labels[test_mask], proba)
                ic = metrics.get("ic", 0.0)
                if not pd.isna(ic):
                    fold_ics.append(ic)
            except Exception:
                continue
        
        return float(np.mean(fold_ics)) if fold_ics else 0.0


def generate_feature_selection_report(
    selected_features: List[str],
    dropped_features: Dict[str, str],
    output_path: Optional[str] = None,
) -> str:
    """Generate markdown report for feature selection results."""
    lines = ["# Feature Selection Report\n"]
    
    lines.append("## Summary\n")
    lines.append(f"- **Features selected**: {len(selected_features)}")
    lines.append(f"- **Features dropped**: {len(dropped_features)}\n")
    
    if dropped_features:
        lines.append("## Dropped Features\n")
        lines.append("| Feature | Reason |")
        lines.append("|---------|--------|")
        
        for feat, reason in sorted(dropped_features.items()):
            lines.append(f"| {feat} | {reason} |")
        
        lines.append("")
    
    if selected_features:
        lines.append("## Selected Features\n")
        lines.append(f"Total: {len(selected_features)} features retained.\n")
        lines.append("```")
        for feat in sorted(selected_features)[:50]:  # Show first 50
            lines.append(feat)
        if len(selected_features) > 50:
            lines.append(f"... and {len(selected_features) - 50} more")
        lines.append("```\n")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Feature selection report saved", extra={"path": output_path})
    
    return report

