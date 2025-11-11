"""Feature layer composition for IntentFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineer:
    """Generate feature blocks for each signal layer.

    Each method is intentionally stubbed with pandas operations and notes on
    how to plug in the real computations (ownership deltas, delivery spikes,
    sentiment scores, technical indicators, etc.).
    """

    feature_blocks: Dict[str, callable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_blocks:
            self.feature_blocks = {
                "technical": self._technical_block,
                "ownership": self._ownership_block,
                "delivery": self._delivery_block,
                "fundamental": self._fundamental_block,
                "narrative": self._narrative_block,
            }

    def build(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for name, builder in self.feature_blocks.items():
            block = builder(dataset.copy())
            if block.empty:
                continue
            block.columns = [f"{name}__{col}" for col in block.columns]
            frames.append(block)
        if frames:
            combined = pd.concat(frames, axis=1)
            return combined.apply(pd.to_numeric, errors="coerce")
        return self._baseline_features(dataset)

    def _baseline_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=dataset.index)
        required = {"close", "volume"}
        if not required.issubset(dataset.columns):
            return output.fillna(0.0)

        if "ticker" in dataset.columns:
            grouped = dataset.groupby("ticker", group_keys=False)
            features = grouped.apply(self._compute_price_block)
        else:
            features = self._compute_price_block(dataset)
        output = features.reindex(dataset.index)
        return output.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def _compute_price_block(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        close = frame["close"]
        volume = frame["volume"]

        result["ret_1d"] = close.pct_change(1)
        result["ema_10"] = close.ewm(span=10, adjust=False).mean()
        result["ema_30"] = close.ewm(span=30, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        result["rsi_14"] = 100 - (100 / (1 + rs))

        rolling = volume.rolling(20)
        result["vol_z"] = (volume - rolling.mean()) / rolling.std()
        return result

    def _technical_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not {"close", "volume"}.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame(index=group.index)
            price = group["close"]
            out["ema_20"] = price.ewm(span=20, adjust=False).mean()
            out["ema_50"] = price.ewm(span=50, adjust=False).mean()
            ema_fast = price.ewm(span=12, adjust=False).mean()
            ema_slow = price.ewm(span=26, adjust=False).mean()
            out["macd"] = ema_fast - ema_slow
            out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
            roll_mean = price.rolling(20).mean()
            roll_std = price.rolling(20).std()
            out["boll_z"] = (price - roll_mean) / roll_std
            out["rsi_14"] = self._compute_price_block(group)["rsi_14"]
            return out

        if "ticker" in dataset.columns:
            features = dataset.groupby("ticker", group_keys=False).apply(compute)
        else:
            features = compute(dataset)
        return features

    def _ownership_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "date", "fii_hold", "dii_hold"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            out["fii_change_5d"] = g["fii_hold"].pct_change(5)
            out["dii_change_5d"] = g["dii_hold"].pct_change(5)
            out["ownership_spread"] = g["fii_hold"] - g["dii_hold"]
            return out

        return dataset.groupby("ticker", group_keys=False).apply(compute)

    def _delivery_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "date", "delivery_ratio"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            rolling = g["delivery_ratio"].rolling(20)
            out["delivery_z"] = (g["delivery_ratio"] - rolling.mean()) / rolling.std()
            out["delivery_spike"] = g["delivery_ratio"] / rolling.mean()
            return out

        return dataset.groupby("ticker", group_keys=False).apply(compute)

    def _fundamental_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "report_date", "eps", "pe"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("report_date")
            out = pd.DataFrame(index=g.index)
            out["eps_yoy"] = g["eps"].pct_change(4)
            out["pe_z"] = (g["pe"] - g["pe"].rolling(4).mean()) / g["pe"].rolling(4).std()
            return out

        return dataset.groupby("ticker", group_keys=False).apply(compute)

    def _narrative_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not {"ticker", "sentiment"}.issubset(dataset.columns):
            return pd.DataFrame()
        agg = (
            dataset.groupby("ticker")["sentiment"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        return pd.DataFrame({"sentiment_mean": agg}, index=agg.index)
