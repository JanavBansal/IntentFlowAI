"""
Market Cap Tiered Models

Train separate models for different market cap segments:
- Large Cap (NIFTY 50): More efficient, lower alpha
- Mid Cap (NIFTY Next 50): Sweet spot for alpha
- Small Cap (rest): Higher alpha, higher risk

Different market cap segments have different characteristics:
- Price discovery efficiency
- Liquidity profiles
- Institutional ownership
- Analyst coverage

Training separate models captures these differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.config.settings import LightGBMConfig
from intentflow_ai.modeling.trainer import LightGBMTrainer
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class MarketCapTier(Enum):
    """Market cap tier classifications."""
    
    LARGE_CAP = "large_cap"      # Top 50 by market cap
    MID_CAP = "mid_cap"          # 51-150 by market cap
    SMALL_CAP = "small_cap"      # 151+ by market cap
    MICRO_CAP = "micro_cap"      # Below threshold


@dataclass
class TierConfig:
    """Configuration for market cap tiers."""
    
    # Market cap thresholds (in INR Crores)
    large_cap_threshold: float = 50_000  # > 50,000 Cr
    mid_cap_threshold: float = 10_000    # 10,000 - 50,000 Cr
    small_cap_threshold: float = 2_000   # 2,000 - 10,000 Cr
    # Below small_cap_threshold = micro_cap
    
    # Alternative: Use rank-based classification
    use_rank_based: bool = True
    large_cap_count: int = 50    # Top 50
    mid_cap_count: int = 100     # Next 100 (51-150)
    # Rest = small cap
    
    # Minimum samples required to train tier-specific model
    min_samples_per_tier: int = 100


@dataclass
class TieredModelConfig:
    """Configuration for tiered modeling."""
    
    tier_config: TierConfig = field(default_factory=TierConfig)
    
    # Model configurations per tier (can customize)
    large_cap_config: Optional[LightGBMConfig] = None
    mid_cap_config: Optional[LightGBMConfig] = None
    small_cap_config: Optional[LightGBMConfig] = None
    
    # Whether to use ensemble within each tier
    use_tier_ensemble: bool = False
    
    # Fallback to overall model if tier has insufficient data
    fallback_to_overall: bool = True


class MarketCapClassifier:
    """
    Classify stocks into market cap tiers.
    """
    
    def __init__(self, config: Optional[TierConfig] = None):
        self.config = config or TierConfig()
    
    def classify(
        self,
        market_caps: pd.Series,
    ) -> pd.Series:
        """
        Classify stocks by market cap.
        
        Args:
            market_caps: Series with ticker index and market cap values
            
        Returns:
            Series with tier classifications
        """
        if self.config.use_rank_based:
            return self._classify_by_rank(market_caps)
        else:
            return self._classify_by_threshold(market_caps)
    
    def _classify_by_rank(self, market_caps: pd.Series) -> pd.Series:
        """Classify by market cap rank."""
        cfg = self.config
        
        # Sort by market cap descending
        sorted_caps = market_caps.sort_values(ascending=False)
        n = len(sorted_caps)
        
        tiers = pd.Series(index=sorted_caps.index, dtype=str)
        
        # Large cap: top N
        large_cap_cutoff = min(cfg.large_cap_count, n)
        tiers.iloc[:large_cap_cutoff] = MarketCapTier.LARGE_CAP.value
        
        # Mid cap: next M
        mid_cap_cutoff = min(cfg.large_cap_count + cfg.mid_cap_count, n)
        tiers.iloc[large_cap_cutoff:mid_cap_cutoff] = MarketCapTier.MID_CAP.value
        
        # Small cap: rest
        tiers.iloc[mid_cap_cutoff:] = MarketCapTier.SMALL_CAP.value
        
        return tiers
    
    def _classify_by_threshold(self, market_caps: pd.Series) -> pd.Series:
        """Classify by market cap threshold."""
        cfg = self.config
        
        def classify_single(mcap):
            if mcap >= cfg.large_cap_threshold:
                return MarketCapTier.LARGE_CAP.value
            elif mcap >= cfg.mid_cap_threshold:
                return MarketCapTier.MID_CAP.value
            elif mcap >= cfg.small_cap_threshold:
                return MarketCapTier.SMALL_CAP.value
            else:
                return MarketCapTier.MICRO_CAP.value
        
        return market_caps.apply(classify_single)
    
    def get_tier_stats(self, tiers: pd.Series) -> Dict[str, int]:
        """Get count statistics per tier."""
        return tiers.value_counts().to_dict()


class TieredModelTrainer:
    """
    Train and manage separate models for each market cap tier.
    
    Usage:
        trainer = TieredModelTrainer()
        
        # Train tiered models
        trainer.train(features_df, labels, market_caps)
        
        # Predict (automatically routes to correct tier model)
        predictions = trainer.predict(features_df, market_caps)
    """
    
    def __init__(self, config: Optional[TieredModelConfig] = None):
        self.config = config or TieredModelConfig()
        self.classifier = MarketCapClassifier(self.config.tier_config)
        
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, LightGBMConfig] = {}
        self.overall_model: Any = None
        self.feature_cols: List[str] = []
    
    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        market_caps: pd.Series,
        base_config: Optional[LightGBMConfig] = None,
    ) -> "TieredModelTrainer":
        """
        Train models for each market cap tier.
        
        Args:
            features: Feature DataFrame (index should match market_caps)
            labels: Target labels
            market_caps: Market cap values per ticker
            base_config: Base LightGBM configuration
            
        Returns:
            Self (for chaining)
        """
        base_config = base_config or LightGBMConfig()
        self.feature_cols = list(features.columns)
        
        # Get tier classifications
        # First, get unique tickers and their market caps
        if "ticker" in features.columns:
            ticker_mcaps = features.groupby("ticker").apply(
                lambda x: market_caps.get(x.name, 0)
            )
        else:
            ticker_mcaps = market_caps
        
        tiers = self.classifier.classify(ticker_mcaps)
        
        logger.info(
            "Training tiered models",
            extra={"tier_distribution": self.classifier.get_tier_stats(tiers)}
        )
        
        # Train overall model first (fallback)
        logger.info("Training overall model...")
        trainer = LightGBMTrainer(base_config)
        self.overall_model = trainer.train(features, labels)
        
        # Map samples to tiers
        if "ticker" in features.columns:
            sample_tiers = features["ticker"].map(tiers)
        else:
            sample_tiers = pd.Series(MarketCapTier.MID_CAP.value, index=features.index)
        
        # Train tier-specific models
        for tier in [MarketCapTier.LARGE_CAP, MarketCapTier.MID_CAP, MarketCapTier.SMALL_CAP]:
            tier_mask = sample_tiers == tier.value
            tier_features = features[tier_mask]
            tier_labels = labels[tier_mask]
            
            if len(tier_features) < self.config.tier_config.min_samples_per_tier:
                logger.warning(
                    f"Insufficient samples for {tier.value}: {len(tier_features)} < {self.config.tier_config.min_samples_per_tier}"
                )
                continue
            
            # Get tier-specific config
            tier_config = self._get_tier_config(tier, base_config)
            self.model_configs[tier.value] = tier_config
            
            logger.info(f"Training {tier.value} model with {len(tier_features)} samples...")
            trainer = LightGBMTrainer(tier_config)
            self.models[tier.value] = trainer.train(tier_features, tier_labels)
        
        logger.info(
            "Tiered model training complete",
            extra={"tiers_trained": list(self.models.keys())}
        )
        
        return self
    
    def _get_tier_config(
        self,
        tier: MarketCapTier,
        base_config: LightGBMConfig,
    ) -> LightGBMConfig:
        """Get configuration for specific tier."""
        from dataclasses import replace
        
        # Use custom config if provided
        if tier == MarketCapTier.LARGE_CAP and self.config.large_cap_config:
            return self.config.large_cap_config
        elif tier == MarketCapTier.MID_CAP and self.config.mid_cap_config:
            return self.config.mid_cap_config
        elif tier == MarketCapTier.SMALL_CAP and self.config.small_cap_config:
            return self.config.small_cap_config
        
        # Tier-specific adjustments
        if tier == MarketCapTier.LARGE_CAP:
            # Large cap: more regularization (efficient market)
            return replace(
                base_config,
                num_leaves=24,
                reg_lambda=2.0,
                min_child_samples=150,
            )
        elif tier == MarketCapTier.SMALL_CAP:
            # Small cap: more flexibility (less efficient)
            return replace(
                base_config,
                num_leaves=48,
                reg_lambda=0.5,
                min_child_samples=50,
            )
        else:
            # Mid cap: use base config
            return base_config
    
    def predict(
        self,
        features: pd.DataFrame,
        market_caps: pd.Series,
    ) -> pd.Series:
        """
        Generate predictions using appropriate tier model.
        
        Args:
            features: Feature DataFrame
            market_caps: Market cap values for tier routing
            
        Returns:
            Predictions Series
        """
        if not self.models and self.overall_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        predictions = pd.Series(index=features.index, dtype=float)
        
        # Get tier classifications
        if "ticker" in features.columns:
            ticker_mcaps = features.groupby("ticker").apply(
                lambda x: market_caps.get(x.name, 0)
            )
            tiers = self.classifier.classify(ticker_mcaps)
            sample_tiers = features["ticker"].map(tiers)
        else:
            sample_tiers = pd.Series(MarketCapTier.MID_CAP.value, index=features.index)
        
        # Predict per tier
        for tier_value, tier_model in self.models.items():
            tier_mask = sample_tiers == tier_value
            if not tier_mask.any():
                continue
            
            tier_features = features.loc[tier_mask, self.feature_cols]
            tier_config = self.model_configs.get(tier_value, LightGBMConfig())
            trainer = LightGBMTrainer(tier_config)
            proba, _ = trainer.predict_with_meta_label(tier_model, tier_features)
            
            predictions.loc[tier_mask] = proba.values
        
        # Use overall model for samples without tier model
        missing_mask = predictions.isna()
        if missing_mask.any() and self.overall_model is not None:
            missing_features = features.loc[missing_mask, self.feature_cols]
            trainer = LightGBMTrainer(LightGBMConfig())
            proba, _ = trainer.predict_with_meta_label(self.overall_model, missing_features)
            predictions.loc[missing_mask] = proba.values
        
        return predictions
    
    def get_feature_importance(self, tier: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance for specified tier or all tiers.
        
        Args:
            tier: Specific tier or None for all
            
        Returns:
            DataFrame with feature importances
        """
        importances = {}
        
        if tier:
            model = self.models.get(tier)
            if model:
                importances[tier] = model.feature_importance(importance_type="gain")
        else:
            # All tiers
            if self.overall_model:
                importances["overall"] = self.overall_model.feature_importance(importance_type="gain")
            
            for tier_value, model in self.models.items():
                importances[tier_value] = model.feature_importance(importance_type="gain")
        
        if not importances:
            return pd.DataFrame()
        
        df = pd.DataFrame(importances, index=self.feature_cols)
        df = df.fillna(0)
        df["average"] = df.mean(axis=1)
        df = df.sort_values("average", ascending=False)
        
        return df


def get_market_cap_tier(
    ticker: str,
    market_cap: float,
    config: Optional[TierConfig] = None,
) -> str:
    """
    Get market cap tier for a single ticker.
    
    Convenience function for quick classification.
    """
    config = config or TierConfig()
    
    if market_cap >= config.large_cap_threshold:
        return MarketCapTier.LARGE_CAP.value
    elif market_cap >= config.mid_cap_threshold:
        return MarketCapTier.MID_CAP.value
    elif market_cap >= config.small_cap_threshold:
        return MarketCapTier.SMALL_CAP.value
    else:
        return MarketCapTier.MICRO_CAP.value


def add_market_cap_features(
    df: pd.DataFrame,
    market_cap_col: str = "market_cap",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Add market cap tier features to DataFrame.
    
    Args:
        df: Input DataFrame
        market_cap_col: Column with market cap values
        ticker_col: Column with ticker names
        
    Returns:
        DataFrame with added tier features
    """
    df = df.copy()
    
    if market_cap_col not in df.columns:
        return df
    
    config = TierConfig()
    
    # Add tier classification
    df["mcap_tier"] = df[market_cap_col].apply(
        lambda x: get_market_cap_tier("", x, config)
    )
    
    # Add tier dummies
    for tier in MarketCapTier:
        df[f"is_{tier.value}"] = (df["mcap_tier"] == tier.value).astype(float)
    
    # Add log market cap (useful feature)
    df["log_market_cap"] = np.log1p(df[market_cap_col])
    
    # Add market cap percentile
    df["mcap_percentile"] = df[market_cap_col].rank(pct=True)
    
    return df
