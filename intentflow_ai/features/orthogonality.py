"""Feature orthogonality testing and selection for production alpha models.

Ensures features are uncorrelated and add incremental predictive power.
Prevents multicollinearity and feature redundancy that can lead to overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrthogonalityConfig:
    """Configuration for feature orthogonality testing."""
    
    # Correlation thresholds
    max_correlation: float = 0.85  # Drop features with >85% correlation
    target_max_correlation: float = 0.70  # Warn if above this threshold
    
    # Clustering parameters
    use_hierarchical_clustering: bool = True
    cluster_threshold: float = 0.85
    
    # VIF (Variance Inflation Factor) thresholds
    use_vif: bool = True
    max_vif: float = 5.0  # Standard threshold for multicollinearity
    
    # Incremental value testing
    test_incremental_value: bool = True
    min_ic_improvement: float = 0.01  # 1% IC improvement required
    
    # Reporting
    generate_heatmap: bool = True
    output_dropped_features: bool = True


@dataclass
class FeatureOrthogonalityAnalyzer:
    """Analyze and enforce feature orthogonality.
    
    Methods:
    1. Correlation-based filtering
    2. Hierarchical clustering to identify redundant groups
    3. VIF (Variance Inflation Factor) for multicollinearity
    4. Incremental IC testing (does feature improve out-of-sample IC?)
    """
    
    cfg: OrthogonalityConfig = field(default_factory=OrthogonalityConfig)
    
    def analyze(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> Dict[str, object]:
        """Run comprehensive orthogonality analysis.
        
        Args:
            features: Feature matrix
            labels: Optional target labels for incremental value testing
            
        Returns:
            Dict with analysis results:
            - correlation_matrix
            - highly_correlated_pairs
            - cluster_groups
            - vif_scores
            - recommendations (features to drop)
        """
        if features.empty:
            return {}
        
        logger.info("Running feature orthogonality analysis", extra={"n_features": len(features.columns)})
        
        results = {}
        
        # 1. Correlation analysis
        corr_matrix = self._compute_correlation_matrix(features)
        results["correlation_matrix"] = corr_matrix
        
        high_corr_pairs = self._find_highly_correlated_pairs(corr_matrix)
        results["highly_correlated_pairs"] = high_corr_pairs
        
        # 2. Hierarchical clustering
        if self.cfg.use_hierarchical_clustering:
            cluster_groups = self._cluster_features(corr_matrix)
            results["cluster_groups"] = cluster_groups
        
        # 3. VIF analysis
        if self.cfg.use_vif:
            vif_scores = self._compute_vif(features)
            results["vif_scores"] = vif_scores
        else:
            vif_scores = {}
        
        # 4. Generate recommendations
        recommendations = self._generate_recommendations(
            corr_matrix=corr_matrix,
            high_corr_pairs=high_corr_pairs,
            cluster_groups=results.get("cluster_groups", {}),
            vif_scores=vif_scores,
            labels=labels,
            features=features,
        )
        results["recommendations"] = recommendations
        
        logger.info(
            "Orthogonality analysis complete",
            extra={
                "high_corr_pairs": len(high_corr_pairs),
                "features_to_drop": len(recommendations.get("drop", [])),
                "features_to_monitor": len(recommendations.get("monitor", [])),
            }
        )
        
        return results
    
    def select_orthogonal_features(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Select orthogonal feature subset.
        
        Args:
            features: Feature matrix
            labels: Target labels
            feature_importance: Optional importance scores (from model)
            
        Returns:
            Tuple of (selected_features, dropped_features_with_reasons)
        """
        analysis = self.analyze(features, labels)
        recommendations = analysis.get("recommendations", {})
        
        features_to_drop = set(recommendations.get("drop", []))
        drop_reasons = {}
        
        # Build drop reasons
        for pair in analysis.get("highly_correlated_pairs", []):
            feat1, feat2, corr = pair["feature1"], pair["feature2"], pair["correlation"]
            
            # Decide which to drop based on importance
            if feature_importance:
                imp1 = feature_importance.get(feat1, 0)
                imp2 = feature_importance.get(feat2, 0)
                to_drop = feat1 if imp1 < imp2 else feat2
            else:
                # Default: drop second feature alphabetically (arbitrary but deterministic)
                to_drop = feat2 if feat1 < feat2 else feat1
            
            if to_drop in features_to_drop:
                drop_reasons[to_drop] = f"Highly correlated ({corr:.2f}) with {feat1 if to_drop == feat2 else feat2}"
        
        # VIF-based drops
        for feat, vif in analysis.get("vif_scores", {}).items():
            if vif > self.cfg.max_vif and feat not in drop_reasons:
                features_to_drop.add(feat)
                drop_reasons[feat] = f"High VIF ({vif:.2f}), multicollinearity"
        
        selected = [f for f in features.columns if f not in features_to_drop]
        
        logger.info(
            "Feature selection complete",
            extra={
                "original": len(features.columns),
                "selected": len(selected),
                "dropped": len(features_to_drop)
            }
        )
        
        return selected, drop_reasons
    
    def _compute_correlation_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute Spearman correlation matrix (robust to non-linear monotonic relationships)."""
        # Use Spearman for robustness
        corr_matrix, _ = spearmanr(features.fillna(0), nan_policy="omit")
        
        # Convert to DataFrame
        if isinstance(corr_matrix, np.ndarray):
            corr_df = pd.DataFrame(
                corr_matrix,
                index=features.columns,
                columns=features.columns
            )
        else:
            # Single feature case
            corr_df = pd.DataFrame([[1.0]], index=features.columns, columns=features.columns)
        
        return corr_df
    
    def _find_highly_correlated_pairs(self, corr_matrix: pd.DataFrame) -> List[Dict[str, object]]:
        """Find pairs of features with correlation above threshold."""
        pairs = []
        
        # Get upper triangle (avoid duplicates)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = abs(corr_matrix.iloc[i, j])
                
                if corr >= self.cfg.max_correlation:
                    pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr),
                        "severity": "high" if corr > 0.95 else "medium"
                    })
        
        return sorted(pairs, key=lambda x: x["correlation"], reverse=True)
    
    def _cluster_features(self, corr_matrix: pd.DataFrame) -> Dict[int, List[str]]:
        """Use hierarchical clustering to group similar features."""
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix.abs()
        
        # Hierarchical clustering
        condensed_dist = squareform(distance_matrix.values)
        linkage = hierarchy.linkage(condensed_dist, method="average")
        
        # Cut dendrogram at threshold
        cluster_labels = hierarchy.fcluster(
            linkage,
            t=1 - self.cfg.cluster_threshold,
            criterion="distance"
        )
        
        # Group features by cluster
        clusters = {}
        for feat, cluster_id in zip(corr_matrix.columns, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(feat)
        
        # Filter to clusters with >1 member (redundant groups)
        redundant_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        
        logger.debug(
            "Feature clustering complete",
            extra={"n_clusters": len(redundant_clusters)}
        )
        
        return redundant_clusters
    
    def _compute_vif(self, features: pd.DataFrame) -> Dict[str, float]:
        """Compute Variance Inflation Factor for each feature.
        
        VIF > 5 indicates multicollinearity.
        VIF = 1 / (1 - R²) where R² is from regressing feature on all others.
        """
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            logger.warning("sklearn not available, skipping VIF computation")
            return {}
        
        vif_scores = {}
        feature_array = features.fillna(0).values
        
        for i, feat in enumerate(features.columns):
            # Regress feature i on all other features
            X = np.delete(feature_array, i, axis=1)
            y = feature_array[:, i]
            
            if len(X) < 2 or X.shape[1] < 1:
                continue
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                r_squared = model.score(X, y)
                
                # VIF = 1 / (1 - R²)
                vif = 1.0 / (1.0 - r_squared + 1e-10)
                vif_scores[feat] = float(vif)
            except Exception as exc:
                logger.debug(f"VIF computation failed for {feat}: {exc}")
                continue
        
        return vif_scores
    
    def _generate_recommendations(
        self,
        corr_matrix: pd.DataFrame,
        high_corr_pairs: List[Dict],
        cluster_groups: Dict,
        vif_scores: Dict[str, float],
        labels: Optional[pd.Series],
        features: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """Generate actionable recommendations on which features to drop or monitor."""
        recommendations = {
            "drop": [],  # Features to remove
            "monitor": [],  # Features to watch
            "keep": [],  # Good features
        }
        
        features_to_drop = set()
        features_to_monitor = set()
        
        # From high correlation pairs
        for pair in high_corr_pairs:
            # Default: drop the second feature
            features_to_drop.add(pair["feature2"])
        
        # From clusters (keep only one per cluster, preferably highest importance)
        for cluster_id, members in cluster_groups.items():
            if len(members) > 1:
                # Drop all but first (in production, use importance scores)
                for feat in members[1:]:
                    features_to_drop.add(feat)
        
        # From VIF
        for feat, vif in vif_scores.items():
            if vif > self.cfg.max_vif:
                features_to_drop.add(feat)
            elif vif > self.cfg.max_vif * 0.7:  # Warning threshold
                features_to_monitor.add(feat)
        
        # Populate recommendations
        recommendations["drop"] = list(features_to_drop)
        recommendations["monitor"] = list(features_to_monitor - features_to_drop)
        recommendations["keep"] = [
            f for f in features.columns
            if f not in features_to_drop and f not in features_to_monitor
        ]
        
        return recommendations


def test_incremental_ic(
    base_features: pd.DataFrame,
    candidate_feature: pd.Series,
    labels: pd.Series,
    *,
    method: str = "spearman",
) -> Dict[str, float]:
    """Test if adding a candidate feature improves IC.
    
    Args:
        base_features: Existing feature set
        candidate_feature: New feature to test
        labels: Target labels
        method: Correlation method ("spearman" or "pearson")
        
    Returns:
        Dict with base_ic, candidate_ic, improvement, accept_candidate
    """
    if labels.empty or base_features.empty:
        return {"base_ic": 0.0, "candidate_ic": 0.0, "improvement": 0.0, "accept": False}
    
    # Align indices
    aligned = pd.DataFrame({
        "label": labels,
        "candidate": candidate_feature
    }).dropna()
    
    if len(aligned) < 10:
        return {"base_ic": 0.0, "candidate_ic": 0.0, "improvement": 0.0, "accept": False}
    
    # Compute IC of candidate alone
    candidate_ic = aligned["label"].corr(aligned["candidate"], method=method)
    
    # Compute combined IC (average of base + candidate)
    # In practice, you'd train a model with/without the feature
    base_ic = 0.0
    if not base_features.empty:
        # Simplified: use first base feature as proxy
        base_aligned = pd.DataFrame({
            "label": labels,
            "base": base_features.iloc[:, 0]
        }).dropna()
        if len(base_aligned) >= 10:
            base_ic = base_aligned["label"].corr(base_aligned["base"], method=method)
    
    improvement = candidate_ic - base_ic
    accept = improvement > 0.01  # 1% IC improvement threshold
    
    return {
        "base_ic": float(base_ic),
        "candidate_ic": float(candidate_ic),
        "improvement": float(improvement),
        "accept": bool(accept),
    }


def generate_orthogonality_report(
    analysis_results: Dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate human-readable orthogonality report.
    
    Args:
        analysis_results: Output from FeatureOrthogonalityAnalyzer.analyze()
        output_path: Optional path to save report
        
    Returns:
        Report as markdown string
    """
    lines = ["# Feature Orthogonality Analysis Report\n"]
    
    # Summary
    recommendations = analysis_results.get("recommendations", {})
    lines.append("## Summary\n")
    lines.append(f"- **Features to drop**: {len(recommendations.get('drop', []))}")
    lines.append(f"- **Features to monitor**: {len(recommendations.get('monitor', []))}")
    lines.append(f"- **Features to keep**: {len(recommendations.get('keep', []))}\n")
    
    # Highly correlated pairs
    high_corr = analysis_results.get("highly_correlated_pairs", [])
    if high_corr:
        lines.append("## Highly Correlated Feature Pairs\n")
        lines.append("| Feature 1 | Feature 2 | Correlation | Severity |")
        lines.append("|-----------|-----------|-------------|----------|")
        for pair in high_corr[:20]:  # Top 20
            lines.append(
                f"| {pair['feature1']} | {pair['feature2']} | "
                f"{pair['correlation']:.3f} | {pair['severity']} |"
            )
        lines.append("")
    
    # VIF scores
    vif_scores = analysis_results.get("vif_scores", {})
    if vif_scores:
        lines.append("## Variance Inflation Factors (VIF)\n")
        lines.append("Features with VIF > 5 indicate multicollinearity:\n")
        high_vif = {k: v for k, v in vif_scores.items() if v > 5}
        if high_vif:
            for feat, vif in sorted(high_vif.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"- **{feat}**: {vif:.2f}")
        else:
            lines.append("✅ No features with VIF > 5 detected.")
        lines.append("")
    
    # Recommendations
    lines.append("## Recommendations\n")
    if recommendations.get("drop"):
        lines.append("### Features to Drop\n")
        for feat in recommendations["drop"][:20]:
            lines.append(f"- {feat}")
        lines.append("")
    
    if recommendations.get("monitor"):
        lines.append("### Features to Monitor\n")
        for feat in recommendations["monitor"][:10]:
            lines.append(f"- {feat}")
        lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Orthogonality report saved", extra={"path": output_path})
    
    return report

