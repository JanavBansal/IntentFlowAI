"""
Monthly Conviction Ranking Engine

Generates monthly conviction rankings for all tickers in the universe.
Integrates:
- Base model predictions
- Probability calibration
- Regime gating
- SHAP-based explanations

Output: Ranked list of tickers with conviction scores and rationale.

Usage:
    ranker = ConvictionRanker(model, calibrator, regime_detector)
    rankings = ranker.generate_monthly_ranking(features_df, current_date)
    ranker.export_ranking(rankings, "rankings/2024-12.csv")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConvictionLevel:
    """Conviction level thresholds."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    @staticmethod
    def from_probability(prob: float) -> str:
        if prob >= 0.65:
            return ConvictionLevel.HIGH
        elif prob >= 0.50:
            return ConvictionLevel.MEDIUM
        else:
            return ConvictionLevel.LOW


@dataclass
class RankingConfig:
    """Configuration for conviction ranking."""
    min_probability_to_rank: float = 0.40  # Minimum P(outperform) to include
    top_n: int = 50                         # Number of top picks to highlight
    shap_top_k: int = 3                     # Top K SHAP features to show
    require_risk_on_regime: bool = True     # Only rank if regime is risk-on
    export_format: str = "csv"              # "csv" or "json"


@dataclass
class TickerRanking:
    """Single ticker ranking result."""
    rank: int
    ticker: str
    sector: str
    probability: float
    conviction: str
    regime: str
    top_drivers: List[str]
    raw_score: float
    
    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "ticker": self.ticker,
            "sector": self.sector,
            "P(Beat30d)": round(self.probability, 4),
            "conviction": self.conviction,
            "regime": self.regime,
            "top_drivers": ", ".join(self.top_drivers),
            "raw_score": round(self.raw_score, 4)
        }


@dataclass
class ConvictionRanker:
    """
    Monthly Conviction Ranking Engine.
    
    Generates ranked lists of tickers by probability of outperformance.
    """
    
    base_model: object                      # Trained ensemble model
    calibrator: Optional[object] = None     # ProbabilityCalibrator
    regime_detector: Optional[object] = None  # RegimeDetector
    config: RankingConfig = field(default_factory=RankingConfig)
    _shap_explainer: Optional[object] = field(default=None, init=False)
    
    def generate_monthly_ranking(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        ranking_date: Optional[str] = None,
        regime_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate monthly conviction rankings.
        
        Args:
            features_df: Feature matrix for all tickers
            prices_df: Price data with ticker, sector info
            ranking_date: Date for ranking (default: latest)
            regime_features: Features for regime detection
            
        Returns:
            DataFrame with ranked tickers
        """
        if ranking_date is None:
            ranking_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Generating monthly ranking for {ranking_date}")
        
        # Check regime
        current_regime = "unknown"
        should_trade = True
        
        if self.regime_detector is not None and regime_features is not None:
            from intentflow_ai.modeling.regime_detector import MarketRegime
            regime_result = self.regime_detector.predict_regime(regime_features)
            current_regime = regime_result.value if hasattr(regime_result, 'value') else str(regime_result)
            should_trade = self.regime_detector.should_trade()
            
            if self.config.require_risk_on_regime and not should_trade:
                logger.warning(f"Regime is {current_regime}, not generating rankings")
                return self._create_empty_ranking(current_regime)
        
        # Get predictions
        X = features_df.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Get raw probabilities
        try:
            raw_probs = self.base_model.predict_proba(X)
            if raw_probs.ndim == 2:
                raw_probs = raw_probs[:, 1]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._create_empty_ranking(current_regime)
        
        # Calibrate probabilities
        if self.calibrator is not None:
            try:
                probs = self.calibrator.predict_proba(X, raw_probs=raw_probs)
            except Exception as e:
                logger.warning(f"Calibration failed: {e}, using raw probs")
                probs = raw_probs
        else:
            probs = raw_probs
        
        # Get SHAP explanations
        shap_drivers = self._get_shap_drivers(X, features_df.columns.tolist())
        
        # Build rankings
        rankings = []
        
        # Get ticker and sector info
        if "ticker" in features_df.columns:
            tickers = features_df["ticker"].values
        else:
            tickers = features_df.index.tolist()
        
        # Get sector mapping
        sector_map = {}
        if "sector" in prices_df.columns and "ticker" in prices_df.columns:
            sector_map = prices_df.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
        
        for i, (ticker, prob, raw_score) in enumerate(zip(tickers, probs, raw_probs)):
            if prob < self.config.min_probability_to_rank:
                continue
            
            sector = sector_map.get(ticker, "Unknown")
            conviction = ConvictionLevel.from_probability(prob)
            drivers = shap_drivers.get(i, ["N/A"])
            
            rankings.append(TickerRanking(
                rank=0,  # Will be set after sorting
                ticker=str(ticker),
                sector=str(sector),
                probability=float(prob),
                conviction=conviction,
                regime=current_regime,
                top_drivers=drivers[:self.config.shap_top_k],
                raw_score=float(raw_score)
            ))
        
        # Sort by probability (descending) and assign ranks
        rankings.sort(key=lambda x: x.probability, reverse=True)
        for i, r in enumerate(rankings):
            r.rank = i + 1
        
        # Convert to DataFrame
        result = pd.DataFrame([r.to_dict() for r in rankings])
        result["ranking_date"] = ranking_date
        
        # DEDUPLICATE: Keep only highest probability per ticker
        if "ticker" in result.columns and len(result) > 0:
            result = result.sort_values(["ticker", "P(Beat30d)"], ascending=[True, False])
            result = result.drop_duplicates(subset=["ticker"], keep="first")
            result = result.sort_values("P(Beat30d)", ascending=False).reset_index(drop=True)
            result["rank"] = range(1, len(result) + 1)
            logger.info(f"Deduplicated to {len(result)} unique tickers")
        
        logger.info(f"Generated rankings for {len(result)} tickers")
        return result
    
    def _get_shap_drivers(
        self,
        X: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[int, List[str]]:
        """Get top SHAP feature drivers for each prediction."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed, skipping explanations")
            return {}
        
        # Only compute for subset if large
        sample_size = min(500, len(X))
        
        try:
            # Use TreeExplainer for tree-based models
            if hasattr(self.base_model, 'models'):
                # For ensemble, use first model (LightGBM)
                base = self.base_model.models.get('lightgbm')
                if base is None:
                    logger.warning("LightGBM model not found in ensemble")
                    return {}
            else:
                base = self.base_model
            
            explainer = shap.TreeExplainer(base)
            
            # Compute SHAP values for sample
            X_sample = X.iloc[:sample_size] if len(X) > sample_size else X
            shap_values = explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Get feature names from numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            drivers = {}
            
            for i in range(len(shap_values)):
                # Get indices of top absolute SHAP values
                abs_shap = np.abs(shap_values[i])
                top_indices = np.argsort(abs_shap)[-self.config.shap_top_k:][::-1]
                
                # Map to feature names
                driver_names = []
                for idx in top_indices:
                    if idx < len(numeric_cols):
                        name = numeric_cols[idx]
                        # Clean up feature names for readability
                        name = name.split('__')[-1].replace('_', ' ').title()
                        sign = "+" if shap_values[i][idx] > 0 else "-"
                        driver_names.append(f"{name}({sign})")
                
                if driver_names:
                    drivers[i] = driver_names
            
            logger.info(f"Computed SHAP drivers for {len(drivers)} samples")
            return drivers
            
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {}
    
    def _create_empty_ranking(self, regime: str) -> pd.DataFrame:
        """Create empty ranking DataFrame with message."""
        return pd.DataFrame({
            "message": [f"No rankings generated. Regime: {regime}"],
            "regime": [regime],
            "ranking_date": [datetime.now().strftime("%Y-%m-%d")]
        })
    
    def export_ranking(
        self,
        rankings: pd.DataFrame,
        output_path: str,
        top_n: Optional[int] = None
    ) -> str:
        """
        Export rankings to file.
        
        Args:
            rankings: Ranking DataFrame
            output_path: Output file path
            top_n: Only export top N (default: all)
            
        Returns:
            Path to exported file
        """
        if top_n is not None:
            rankings = rankings.head(top_n)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.export_format == "json":
            rankings.to_json(output_path, orient="records", indent=2)
        else:
            rankings.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(rankings)} rankings to {output_path}")
        return str(output_path)
    
    def generate_ranking_report(
        self,
        rankings: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a formatted markdown report of rankings.
        
        Args:
            rankings: Ranking DataFrame
            output_path: Optional path to save report
            
        Returns:
            Markdown report string
        """
        if rankings.empty or "message" in rankings.columns:
            return f"# Monthly Conviction Ranking\n\n⚠️ No rankings generated.\n"
        
        date = rankings["ranking_date"].iloc[0] if "ranking_date" in rankings.columns else "N/A"
        regime = rankings["regime"].iloc[0] if "regime" in rankings.columns else "unknown"
        
        report = f"""# Monthly Conviction Ranking

**Date:** {date}
**Regime:** {regime}
**Total Ranked:** {len(rankings)}

## Top 20 Picks

| Rank | Ticker | Sector | P(Beat) | Conviction | Top Drivers |
|------|--------|--------|---------|------------|-------------|
"""
        
        for _, row in rankings.head(20).iterrows():
            report += f"| {row['rank']} | {row['ticker']} | {row['sector']} | {row['P(Beat30d)']:.2%} | {row['conviction']} | {row['top_drivers']} |\n"
        
        # Add sector distribution
        report += "\n## Sector Distribution (Top 50)\n\n"
        sector_counts = rankings.head(50)["sector"].value_counts()
        for sector, count in sector_counts.items():
            report += f"- **{sector}:** {count}\n"
        
        # Add conviction breakdown
        report += "\n## Conviction Breakdown\n\n"
        conv_counts = rankings["conviction"].value_counts()
        for conv, count in conv_counts.items():
            pct = count / len(rankings) * 100
            report += f"- **{conv.capitalize()}:** {count} ({pct:.1f}%)\n"
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Saved ranking report to {output_path}")
        
        return report


def test_conviction_ranker():
    """Test the conviction ranker with sample data."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample model
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create sample features for ranking
    tickers = [f"STOCK{i}" for i in range(100)]
    feature_names = [f"feature_{i}" for i in range(50)]
    
    X_rank = pd.DataFrame(np.random.randn(100, 50), columns=feature_names)
    X_rank["ticker"] = tickers
    
    prices_df = pd.DataFrame({
        "ticker": tickers,
        "sector": np.random.choice(["IT", "Banks", "Pharma", "Auto", "Energy"], 100)
    })
    
    # Create ranker
    ranker = ConvictionRanker(base_model=model)
    
    # Generate rankings
    rankings = ranker.generate_monthly_ranking(X_rank, prices_df)
    
    print(f"Rankings shape: {rankings.shape}")
    print(rankings.head(10))
    
    # Generate report
    report = ranker.generate_ranking_report(rankings)
    print("\n" + report[:1000])
    
    return rankings


if __name__ == "__main__":
    test_conviction_ranker()
