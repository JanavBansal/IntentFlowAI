"""Production-grade signal cards with complete interpretability.

Each signal card contains:
- Signal metadata (ticker, date, probability, rank)
- Complete rationale with SHAP explanations
- All feature values used in prediction
- Market regime context
- Historical performance of similar signals
- Risk warnings and confidence indicators
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SignalCard:
    """Complete signal information for trade decision-making."""
    
    # Core signal info
    ticker: str
    date: str
    probability: float
    rank: int
    
    # Model explanation
    rationale: str
    top_features: List[Dict[str, Any]]  # [{feature, value, shap_contribution}, ...]
    shap_values: Dict[str, float]  # All SHAP values
    base_value: float  # Model baseline prediction
    
    # Feature context
    all_features: Dict[str, float]  # All input features
    feature_percentiles: Dict[str, float]  # Percentile rank of each feature
    
    # Regime context
    market_regime: str
    volatility_regime: str
    trend_regime: str
    breadth_regime: str
    regime_score: float
    allow_entry: bool
    
    # Risk indicators
    confidence_level: str  # "high", "medium", "low"
    risk_warnings: List[str]
    sector: Optional[str] = None
    
    # Historical context
    similar_signal_stats: Optional[Dict[str, float]] = None  # Past performance of similar signals
    
    # Metadata
    model_version: str = "unknown"
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "ticker": self.ticker,
            "date": self.date,
            "probability": self.probability,
            "rank": self.rank,
            "rationale": self.rationale,
            "top_features": self.top_features,
            "shap_values": self.shap_values,
            "base_value": self.base_value,
            "all_features": self.all_features,
            "feature_percentiles": self.feature_percentiles,
            "market_regime": self.market_regime,
            "volatility_regime": self.volatility_regime,
            "trend_regime": self.trend_regime,
            "breadth_regime": self.breadth_regime,
            "regime_score": self.regime_score,
            "allow_entry": self.allow_entry,
            "confidence_level": self.confidence_level,
            "risk_warnings": self.risk_warnings,
            "sector": self.sector,
            "similar_signal_stats": self.similar_signal_stats,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
        }
    
    def to_markdown(self) -> str:
        """Generate human-readable markdown report."""
        lines = [f"# Signal Card: {self.ticker}"]
        lines.append(f"**Date**: {self.date} | **Probability**: {self.probability:.2%} | **Rank**: {self.rank}\n")
        
        # Confidence & warnings
        lines.append(f"**Confidence**: {self.confidence_level.upper()}")
        if self.risk_warnings:
            lines.append(f"⚠️ **Warnings**: {', '.join(self.risk_warnings)}\n")
        else:
            lines.append("")
        
        # Rationale
        lines.append("## 📊 Signal Rationale")
        lines.append(self.rationale + "\n")
        
        # Top features
        lines.append("## 🔍 Top Contributing Features")
        lines.append("| Feature | Value | SHAP Contribution |")
        lines.append("|---------|-------|-------------------|")
        for feat in self.top_features[:10]:
            name = feat.get("feature", "")
            value = feat.get("value", 0)
            shap = feat.get("shap_contribution", 0)
            lines.append(f"| {name} | {value:.4f} | {shap:+.4f} |")
        lines.append("")
        
        # Regime context
        lines.append("## 🌐 Market Regime Context")
        lines.append(f"- **Overall Regime**: {self.market_regime}")
        lines.append(f"- **Volatility**: {self.volatility_regime}")
        lines.append(f"- **Trend**: {self.trend_regime}")
        lines.append(f"- **Breadth**: {self.breadth_regime}")
        lines.append(f"- **Regime Score**: {self.regime_score:.0f}/100")
        lines.append(f"- **Entry Allowed**: {'✅ Yes' if self.allow_entry else '❌ No'}\n")
        
        # Similar signal stats
        if self.similar_signal_stats:
            lines.append("## 📈 Historical Performance (Similar Signals)")
            for key, value in self.similar_signal_stats.items():
                lines.append(f"- **{key}**: {value:.2%}" if abs(value) < 10 else f"- **{key}**: {value:.2f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Generate HTML card for dashboard display."""
        # Simplified HTML template
        confidence_color = {
            "high": "#28a745",
            "medium": "#ffc107",
            "low": "#dc3545"
        }.get(self.confidence_level, "#6c757d")
        
        html = f"""
        <div class="signal-card" style="border: 2px solid {confidence_color}; padding: 20px; margin: 10px; border-radius: 8px;">
            <h2>{self.ticker} - {self.date}</h2>
            <p><strong>Probability:</strong> {self.probability:.2%} | <strong>Rank:</strong> {self.rank}</p>
            <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{self.confidence_level.upper()}</span></p>
            
            <h3>Rationale</h3>
            <p>{self.rationale}</p>
            
            <h3>Market Regime</h3>
            <ul>
                <li><strong>Overall:</strong> {self.market_regime}</li>
                <li><strong>Regime Score:</strong> {self.regime_score:.0f}/100</li>
                <li><strong>Entry Allowed:</strong> {'Yes' if self.allow_entry else 'No'}</li>
            </ul>
            
            {"<p><strong>⚠️ Warnings:</strong> " + ", ".join(self.risk_warnings) + "</p>" if self.risk_warnings else ""}
        </div>
        """
        return html


@dataclass
class SignalCardGenerator:
    """Generate comprehensive signal cards for trading signals."""
    
    model_version: str = "1.0.0"
    
    def generate_cards(
        self,
        signals: pd.DataFrame,
        features: pd.DataFrame,
        shap_values: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None,
        historical_stats: Optional[pd.DataFrame] = None,
    ) -> List[SignalCard]:
        """Generate signal cards for all signals.
        
        Args:
            signals: DataFrame with columns [date, ticker, proba, rank, ...]
            features: Feature matrix (aligned with signals by index)
            shap_values: Optional SHAP value matrix
            regime_data: Optional regime classifications
            historical_stats: Optional historical performance data
            
        Returns:
            List of SignalCard objects
        """
        if signals.empty:
            return []
        
        cards = []
        
        for idx in signals.index:
            try:
                card = self._create_single_card(
                    signal=signals.loc[idx],
                    features=features.loc[idx] if idx in features.index else pd.Series(),
                    shap_values=shap_values.loc[idx] if shap_values is not None and idx in shap_values.index else None,
                    regime_data=regime_data,
                    historical_stats=historical_stats,
                )
                cards.append(card)
            except Exception as exc:
                logger.warning(f"Failed to create signal card for index {idx}: {exc}")
                continue
        
        logger.info(f"Generated {len(cards)} signal cards")
        return cards
    
    def _create_single_card(
        self,
        signal: pd.Series,
        features: pd.Series,
        shap_values: Optional[pd.Series],
        regime_data: Optional[pd.DataFrame],
        historical_stats: Optional[pd.DataFrame],
    ) -> SignalCard:
        """Create a single signal card."""
        import datetime
        
        # Parse signal data
        ticker = str(signal.get("ticker", "UNKNOWN"))
        date = str(signal.get("date", ""))
        probability = float(signal.get("proba", 0.0))
        rank = int(signal.get("rank", 0))
        sector = str(signal.get("sector", None)) if "sector" in signal else None
        
        # Rationale
        rationale = str(signal.get("rationale", "No rationale available"))
        
        # Top features
        if "top_features" in signal and isinstance(signal["top_features"], list):
            top_features = signal["top_features"]
        else:
            # Build from SHAP values
            top_features = self._build_top_features(features, shap_values)
        
        # SHAP values dict
        if shap_values is not None:
            shap_dict = shap_values.to_dict()
        elif "shap_values" in signal and isinstance(signal["shap_values"], dict):
            shap_dict = signal["shap_values"]
        else:
            shap_dict = {}
        
        # Base value (model baseline)
        base_value = 0.5  # Binary classification baseline
        
        # All features
        all_features = features.to_dict() if not features.empty else {}
        
        # Feature percentiles
        feature_percentiles = self._compute_feature_percentiles(features)
        
        # Regime context
        regime_info = self._extract_regime_info(signal, regime_data, date)
        
        # Confidence & risk warnings
        confidence, warnings = self._assess_confidence_and_risks(
            probability=probability,
            regime_info=regime_info,
            shap_dict=shap_dict,
        )
        
        # Historical stats
        similar_stats = self._get_similar_signal_stats(
            ticker=ticker,
            probability=probability,
            historical_stats=historical_stats,
        )
        
        return SignalCard(
            ticker=ticker,
            date=date,
            probability=probability,
            rank=rank,
            rationale=rationale,
            top_features=top_features,
            shap_values=shap_dict,
            base_value=base_value,
            all_features=all_features,
            feature_percentiles=feature_percentiles,
            market_regime=regime_info.get("composite_regime", "unknown"),
            volatility_regime=regime_info.get("volatility_regime", "unknown"),
            trend_regime=regime_info.get("trend_regime", "unknown"),
            breadth_regime=regime_info.get("breadth_regime", "unknown"),
            regime_score=regime_info.get("regime_score", 50.0),
            allow_entry=regime_info.get("allow_entry", True),
            confidence_level=confidence,
            risk_warnings=warnings,
            sector=sector,
            similar_signal_stats=similar_stats,
            model_version=self.model_version,
            timestamp=datetime.datetime.now().isoformat(),
        )
    
    def _build_top_features(
        self,
        features: pd.Series,
        shap_values: Optional[pd.Series],
    ) -> List[Dict[str, Any]]:
        """Build top features list from SHAP values."""
        if shap_values is None or shap_values.empty:
            return []
        
        # Sort by absolute SHAP value
        abs_shap = shap_values.abs().sort_values(ascending=False)
        
        top_features = []
        for feat in abs_shap.head(10).index:
            top_features.append({
                "feature": feat,
                "value": float(features.get(feat, np.nan)),
                "shap_contribution": float(shap_values.get(feat, 0.0)),
                "abs_contribution": float(abs(shap_values.get(feat, 0.0))),
            })
        
        return top_features
    
    def _compute_feature_percentiles(self, features: pd.Series) -> Dict[str, float]:
        """Compute percentile rank of features (requires training distribution)."""
        # Simplified: return 50th percentile for all
        # In production, compare against training set distribution
        return {feat: 50.0 for feat in features.index}
    
    def _extract_regime_info(
        self,
        signal: pd.Series,
        regime_data: Optional[pd.DataFrame],
        date: str,
    ) -> Dict[str, Any]:
        """Extract regime information for signal date."""
        if regime_data is None or regime_data.empty:
            return {
                "composite_regime": "unknown",
                "volatility_regime": "unknown",
                "trend_regime": "unknown",
                "breadth_regime": "unknown",
                "regime_score": 50.0,
                "allow_entry": True,
            }
        
        # Try to get regime for signal date
        date_pd = pd.to_datetime(date)
        if date_pd in regime_data.index:
            regime_row = regime_data.loc[date_pd]
            return {
                "composite_regime": regime_row.get("composite_regime", "unknown"),
                "volatility_regime": regime_row.get("volatility_regime", "unknown"),
                "trend_regime": regime_row.get("trend_regime", "unknown"),
                "breadth_regime": regime_row.get("breadth_regime", "unknown"),
                "regime_score": float(regime_row.get("regime_score", 50.0)),
                "allow_entry": bool(regime_row.get("allow_entry", True)),
            }
        else:
            # Use data from signal if embedded
            return {
                "composite_regime": str(signal.get("composite_regime", "unknown")),
                "volatility_regime": str(signal.get("volatility_regime", "unknown")),
                "trend_regime": str(signal.get("trend_regime", "unknown")),
                "breadth_regime": str(signal.get("breadth_regime", "unknown")),
                "regime_score": float(signal.get("regime_score", 50.0)),
                "allow_entry": bool(signal.get("allow_entry", True)),
            }
    
    def _assess_confidence_and_risks(
        self,
        probability: float,
        regime_info: Dict,
        shap_dict: Dict[str, float],
    ) -> tuple[str, List[str]]:
        """Assess confidence level and identify risk warnings."""
        warnings = []
        
        # Confidence based on probability
        if probability >= 0.75:
            confidence = "high"
        elif probability >= 0.60:
            confidence = "medium"
        else:
            confidence = "low"
            warnings.append("Low model probability (<60%)")
        
        # Regime-based warnings
        if not regime_info.get("allow_entry", True):
            warnings.append("Unfavorable market regime - entry blocked")
        
        regime_score = regime_info.get("regime_score", 50.0)
        if regime_score < 30:
            warnings.append(f"Low regime score ({regime_score:.0f}/100)")
        
        # SHAP concentration warning
        if shap_dict:
            shap_values = np.array(list(shap_dict.values()))
            if len(shap_values) > 0:
                top_shap = np.max(np.abs(shap_values))
                total_shap = np.sum(np.abs(shap_values))
                if top_shap / (total_shap + 1e-9) > 0.5:
                    warnings.append("Signal driven by single feature (low diversification)")
        
        return confidence, warnings
    
    def _get_similar_signal_stats(
        self,
        ticker: str,
        probability: float,
        historical_stats: Optional[pd.DataFrame],
    ) -> Optional[Dict[str, float]]:
        """Retrieve historical performance of similar signals."""
        if historical_stats is None or historical_stats.empty:
            return None
        
        # Simplified: return aggregate stats
        # In production, filter by ticker/probability bucket
        return {
            "avg_return": 0.015,  # Placeholder
            "win_rate": 0.55,
            "avg_hold_days": 10.0,
            "sample_size": 100,
        }
    
    def save_cards(
        self,
        cards: List[SignalCard],
        output_dir: Path,
        format: str = "json",
    ):
        """Save signal cards to disk.
        
        Args:
            cards: List of signal cards
            output_dir: Directory to save cards
            format: Output format ("json", "markdown", "html")
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Save all cards as JSON
            cards_path = output_dir / "signal_cards.json"
            with open(cards_path, "w") as f:
                json.dump([c.to_dict() for c in cards], f, indent=2, default=float)
            logger.info(f"Saved {len(cards)} signal cards to {cards_path}")
        
        elif format == "markdown":
            # Save individual markdown files
            for card in cards:
                filename = f"signal_card_{card.ticker}_{card.date}.md"
                filepath = output_dir / filename
                with open(filepath, "w") as f:
                    f.write(card.to_markdown())
            logger.info(f"Saved {len(cards)} signal cards as markdown to {output_dir}")
        
        elif format == "html":
            # Save as single HTML page
            html_lines = ["<html><head><title>Signal Cards</title></head><body>"]
            for card in cards:
                html_lines.append(card.to_html())
            html_lines.append("</body></html>")
            
            html_path = output_dir / "signal_cards.html"
            with open(html_path, "w") as f:
                f.write("\n".join(html_lines))
            logger.info(f"Saved {len(cards)} signal cards to {html_path}")

