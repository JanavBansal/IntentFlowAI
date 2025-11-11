"""Feature layer composition for IntentFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class FeatureEngineer:
    """Generate feature blocks for each signal layer.

    Each method is intentionally stubbed with pandas operations and notes on
    how to plug in the real computations (ownership deltas, delivery spikes,
    sentiment scores, technical indicators, etc.).
    """

    feature_blocks: Dict[str, callable] = field(default_factory=dict)

    def build(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for name, builder in self.feature_blocks.items():
            block = builder(dataset.copy())
            block.columns = [f"{name}__{col}" for col in block.columns]
            frames.append(block)
        if frames:
            return pd.concat(frames, axis=1)
        # Minimal fallback uses simple momentum stats as scaffolding
        return self._baseline_features(dataset)

    def _baseline_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=dataset.index)
        if {"close", "volume"}.issubset(dataset.columns):
            output["returns_5d"] = dataset["close"].pct_change(5)
            output["volume_z"] = (
                (dataset["volume"] - dataset["volume"].rolling(20).mean())
                / dataset["volume"].rolling(20).std()
            )
        return output.fillna(0.0)
