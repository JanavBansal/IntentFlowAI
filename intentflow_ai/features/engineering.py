"""Feature layer composition for IntentFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from intentflow_ai.config.settings import settings
from intentflow_ai.data.universe import load_universe
from intentflow_ai.modeling.regimes import RegimeClassifier


@lru_cache(maxsize=1)
def _sector_lookup() -> pd.Series:
    try:
        universe = load_universe(settings.path(settings.universe_file))
        return universe.set_index("ticker_nse")["sector"]
    except Exception:
        return pd.Series(dtype="string")


@dataclass
class FeatureEngineer:
    """Generate feature blocks for each signal layer.

    Each method is intentionally stubbed with pandas operations and notes on
    how to plug in the real computations (ownership deltas, delivery spikes,
    sentiment scores, technical indicators, etc.).
    """

    feature_blocks: Dict[str, callable] = field(default_factory=dict)
    regime_classifier: RegimeClassifier = field(default_factory=RegimeClassifier)

    def __post_init__(self) -> None:
        if not self.feature_blocks:
            self.feature_blocks = {
                "technical": self._technical_block,
                "momentum": self._momentum_block,
                "volatility": self._volatility_block,
                "atr": self._atr_block,
                "turnover": self._turnover_block,
                "ownership": self._ownership_block,
                "delivery": self._delivery_block,
                "fundamental": self._fundamental_block,
                "narrative": self._narrative_block,
                "sector_relative": self._sector_relative_block,
                "regime": self._regime_block,
            }

    @staticmethod
    def _group_apply(grouped: pd.core.groupby.DataFrameGroupBy, func) -> pd.DataFrame:
        try:
            return grouped.apply(func, include_groups=False)
        except TypeError:
            return grouped.apply(func)

    def build(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        if "date" in dataset.columns:
            dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        if "sector" not in dataset.columns or dataset["sector"].isna().any():
            lookup = _sector_lookup()
            if not lookup.empty and "ticker" in dataset.columns:
                dataset["sector"] = dataset.get("sector", pd.Series(index=dataset.index, dtype="string"))
                dataset["sector"] = dataset["sector"].fillna(dataset["ticker"].map(lookup))
        frames: List[pd.DataFrame] = []
        for name, builder in self.feature_blocks.items():
            block = builder(dataset.copy())
            if block.empty:
                continue
            block.columns = [f"{name}__{col}" for col in block.columns]
            frames.append(block)
        if frames:
            combined = pd.concat(frames, axis=1)
            combined = combined.apply(pd.to_numeric, errors="coerce")
            combined = combined.replace([np.inf, -np.inf], np.nan)
            return combined.fillna(0.0)
        return self._baseline_features(dataset)

    def _baseline_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=dataset.index)
        required = {"close", "volume"}
        if not required.issubset(dataset.columns):
            return output.fillna(0.0)

        if "ticker" in dataset.columns:
            grouped = dataset.groupby("ticker", group_keys=False)
            features = self._group_apply(grouped, self._compute_price_block)
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
            features = self._group_apply(dataset.groupby("ticker", group_keys=False), compute)
        else:
            features = compute(dataset)
        return features

    def _momentum_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            price = g["close"]
            out["ret_3d"] = price.pct_change(3)
            out["ret_5d"] = price.pct_change(5)
            out["ret_10d"] = price.pct_change(10)
            out["ret_20d"] = price.pct_change(20)
            out["momentum_ratio_10_30"] = price.rolling(10).mean() / price.rolling(30).mean() - 1.0
            out["pct_from_120d_high"] = price / price.rolling(120, min_periods=20).max() - 1.0
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _volatility_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            rets = g["close"].pct_change()
            out["vol_5d"] = rets.rolling(5).std()
            out["vol_10d"] = rets.rolling(10).std()
            out["vol_20d"] = rets.rolling(20).std()
            out["downside_vol_10d"] = rets.clip(upper=0).rolling(10).std()
            out["vol_ratio_short_long"] = out["vol_5d"] / (out["vol_20d"] + 1e-9)
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _atr_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "date", "high", "low", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            high = g["high"]
            low = g["low"]
            close = g["close"]
            prev_close = close.shift(1)
            tr = pd.concat(
                [
                    (high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            out["atr_14"] = tr.rolling(14).mean()
            out["atr_pct_14"] = out["atr_14"] / (close.replace(0, np.nan))
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _turnover_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "date", "volume"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            volume = g["volume"]
            rolling_20 = volume.rolling(20)
            out["turnover_z_20"] = (volume - rolling_20.mean()) / rolling_20.std()
            out["turnover_trend_20"] = volume.pct_change(20)
            out["volume_ratio_5_20"] = volume.rolling(5).mean() / (rolling_20.mean() + 1e-9)
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

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
            inst = g["fii_hold"] + g["dii_hold"]
            out["ownership_trend_20d"] = inst.pct_change(20)
            out["fii_to_dii_ratio"] = g["fii_hold"] / (g["dii_hold"].replace(0, np.nan))
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

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
            short = g["delivery_ratio"].rolling(5)
            out["delivery_trend_5d"] = g["delivery_ratio"].pct_change(5)
            out["delivery_ratio_5_20"] = short.mean() / (rolling.mean() + 1e-9)
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

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

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _narrative_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not {"ticker", "sentiment"}.issubset(dataset.columns):
            return pd.DataFrame()
        agg = (
            dataset.groupby("ticker")["sentiment"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        sentiment_change = dataset.groupby("ticker")["sentiment"].diff()
        return pd.DataFrame({"sentiment_mean": agg, "sentiment_change": sentiment_change}, index=agg.index)

    def _sector_relative_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"ticker", "sector", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        frame = dataset[["ticker", "sector", "date", "close"]].copy()
        frame["ret_5d"] = (
            dataset.groupby("ticker", group_keys=False)["close"].pct_change(5)
        )
        sector_group = frame.groupby(["date", "sector"], group_keys=False)["ret_5d"]
        mean = sector_group.transform("mean")
        std = sector_group.transform("std").replace(0, np.nan)
        ranks = sector_group.rank(pct=True)
        sector_close = dataset.groupby(["date", "sector"], group_keys=False)["close"].transform("mean")

        result = pd.DataFrame(index=dataset.index)
        result["sector_ret_z"] = (frame["ret_5d"] - mean) / std
        result["sector_rank_pct"] = ranks
        result["sector_rel_close"] = dataset["close"] / sector_close - 1.0
        return result

    def _regime_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()
        market = dataset.groupby("date")["close"].mean().sort_index()
        regime_map = self.regime_classifier.infer(market).ffill().bfill()
        vol_series = market.pct_change().rolling(20).std()
        regime_frame = regime_map.rename("market_regime").reset_index()
        vol_frame = vol_series.rename("market_vol_20d").reset_index()

        merged = pd.DataFrame({"date": dataset["date"]}, index=dataset.index)
        merged = merged.merge(regime_frame, on="date", how="left")
        merged = merged.merge(vol_frame, on="date", how="left")

        out = pd.DataFrame(index=dataset.index)
        out["regime_is_bull"] = (merged["market_regime"] == "bull").astype(float)
        out["regime_is_bear"] = (merged["market_regime"] == "bear").astype(float)
        out["market_vol_20d"] = merged["market_vol_20d"]
        return out
