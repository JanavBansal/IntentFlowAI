"""Market regime classification utilities."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class RegimeClassifier:
    """Simple bull/bear filter based on realized volatility.

    Real implementation can incorporate macro indicators, breadth stats, and
    volatility term structure. This placeholder illustrates where such logic
    lives so the LightGBM trainer can condition on regime states.
    """

    vol_threshold: float = 0.02

    def infer(self, price_series: pd.Series) -> pd.Series:
        realized_vol = price_series.pct_change().rolling(20).std()
        return (realized_vol < self.vol_threshold).map({True: "bull", False: "bear"})
