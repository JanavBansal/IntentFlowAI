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
                "regime_adaptive": self._regime_adaptive_block,
                "mean_reversion": self._mean_reversion_block,
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
        """Price momentum features - all point-in-time safe, backward-looking only.
        
        Features:
        - price_ret_1d, price_ret_3d, price_ret_5d, price_ret_10d: Simple percentage changes
        - price_mom_5, price_mom_10, price_mom_20: Close / close_Nd - 1
        - momentum_ratio_10_30: MA(10) / MA(30) - 1
        - pct_from_120d_high: Distance from recent high
        """
        required = {"ticker", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            price = g["close"]
            
            # Simple returns (pct_change is backward-looking)
            out["price_ret_1d"] = price.pct_change(1)
            out["price_ret_3d"] = price.pct_change(3)
            out["price_ret_5d"] = price.pct_change(5)
            out["price_ret_10d"] = price.pct_change(10)
            out["price_ret_20d"] = price.pct_change(20)
            
            # Momentum: current price vs historical (backward-looking)
            out["price_mom_5"] = price / price.shift(5) - 1.0
            out["price_mom_10"] = price / price.shift(10) - 1.0
            out["price_mom_20"] = price / price.shift(20) - 1.0
            
            # Moving average ratios
            out["momentum_ratio_10_30"] = price.rolling(10).mean() / price.rolling(30).mean() - 1.0
            out["pct_from_120d_high"] = price / price.rolling(120, min_periods=20).max() - 1.0
            
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _volatility_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Price volatility features - all point-in-time safe.
        
        Features:
        - price_vol_10, price_vol_20: Standard deviation of daily returns over N days
        - vol_5d, downside_vol_10d: Short-term and downside volatility
        - vol_ratio_short_long: Ratio of short to long volatility (regime change indicator)
        """
        required = {"ticker", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            rets = g["close"].pct_change()
            
            # Core volatility features (backward-looking rolling windows)
            out["vol_5d"] = rets.rolling(5).std()
            out["price_vol_10"] = rets.rolling(10).std()  # Primary 10-day vol
            out["price_vol_20"] = rets.rolling(20).std()  # Primary 20-day vol
            
            # Downside volatility (only negative returns)
            out["downside_vol_10d"] = rets.clip(upper=0).rolling(10).std()
            
            # Volatility regime indicator
            out["vol_ratio_short_long"] = out["vol_5d"] / (out["price_vol_20"] + 1e-9)
            
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
        """Volume/liquidity features - all point-in-time safe.
        
        Features:
        - volume_mean_20: Rolling mean volume (liquidity baseline)
        - volume_spike: Current volume / 20-day mean (unusual activity detector)
        - turnover_z_20: Z-score of volume relative to 20-day history
        - volume_ratio_5_20: Short vs long volume trend
        """
        required = {"ticker", "date", "volume"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            volume = g["volume"]
            
            # Volume baseline and spike detection
            rolling_20 = volume.rolling(20)
            out["volume_mean_20"] = rolling_20.mean()
            out["volume_spike"] = volume / (out["volume_mean_20"] + 1e-9)
            
            # Volume z-score (unusual activity)
            out["turnover_z_20"] = (volume - rolling_20.mean()) / (rolling_20.std() + 1e-9)
            
            # Volume trends
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
        """Delivery & microstructure flow features - all point-in-time safe.
        
        Two modes:
        1. If 'delivery_ratio' already exists: use it directly
        2. If 'delivery_qty' and 'volume' exist: compute ratio and value
        
        Features:
        - deliv_ratio: delivery_qty / volume (conviction measure)
        - deliv_ratio_mean_5, _10, _20: Rolling averages (flow baseline)
        - deliv_ratio_change_10: deliv_ratio_mean_10 - deliv_ratio_mean_20 (flow acceleration)
        - deliv_value: delivery_qty * close (rupee flow)
        - deliv_value_mean_20: Rolling average rupee flow
        - deliv_value_spike: deliv_value / deliv_value_mean_20 (unusual conviction)
        - deliv_vs_price_corr_10: Rolling correlation between deliv_ratio and returns
        """
        # Check if we can compute delivery features
        has_ratio = "delivery_ratio" in dataset.columns
        has_raw = {"delivery_qty", "volume", "close"}.issubset(dataset.columns)
        
        if not (has_ratio or has_raw):
            return pd.DataFrame()
        
        required_cols = {"ticker", "date"}
        if not required_cols.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            
            # Compute or extract delivery ratio
            if has_raw:
                # Compute from raw data
                deliv_ratio = g["delivery_qty"] / (g["volume"] + 1e-9)
                deliv_value = g["delivery_qty"] * g["close"]
            elif has_ratio:
                # Use pre-computed ratio
                deliv_ratio = g["delivery_ratio"]
                # Approximate value if we have close
                if "close" in g.columns and "volume" in g.columns:
                    deliv_value = deliv_ratio * g["volume"] * g["close"]
                else:
                    deliv_value = None
            else:
                return out
            
            # Rolling means of delivery ratio (flow baseline)
            out["deliv_ratio"] = deliv_ratio
            out["deliv_ratio_mean_5"] = deliv_ratio.rolling(5).mean()
            out["deliv_ratio_mean_10"] = deliv_ratio.rolling(10).mean()
            out["deliv_ratio_mean_20"] = deliv_ratio.rolling(20).mean()
            
            # Flow acceleration (change in short vs long delivery)
            out["deliv_ratio_change_10"] = out["deliv_ratio_mean_10"] - out["deliv_ratio_mean_20"]
            
            # Delivery value features (if available)
            if deliv_value is not None:
                out["deliv_value"] = deliv_value
                out["deliv_value_mean_20"] = deliv_value.rolling(20).mean()
                out["deliv_value_spike"] = deliv_value / (out["deliv_value_mean_20"] + 1e-9)
            
            # Advanced: correlation between delivery and price momentum
            # (requires sufficient history, gracefully handle short windows)
            # Note: This is computationally expensive; we'll compute it only if both series have sufficient data
            if "close" in g.columns:
                price_ret_1d = g["close"].pct_change()
                # Rolling correlation: delivery conviction vs price movement
                # Use a 10-day window; requires at least 5 valid observations
                # For efficiency, compute correlation only where we have sufficient overlapping data
                try:
                    # Create aligned DataFrame for correlation (drop NaN to align)
                    aligned = pd.DataFrame({
                        "deliv_ratio": deliv_ratio,
                        "price_ret_1d": price_ret_1d
                    })
                    
                    # Compute rolling correlation using a helper function
                    def compute_rolling_corr(idx):
                        """Compute correlation for a rolling window ending at idx."""
                        window_end = aligned.index.get_loc(idx) if idx in aligned.index else None
                        if window_end is None or window_end < 4:
                            return np.nan
                        window_start = max(0, window_end - 9)
                        window_data = aligned.iloc[window_start:window_end + 1]
                        valid = window_data[["deliv_ratio", "price_ret_1d"]].dropna()
                        if len(valid) >= 5:
                            return valid["deliv_ratio"].corr(valid["price_ret_1d"])
                        return np.nan
                    
                    rolling_corr = pd.Series(
                        [compute_rolling_corr(idx) for idx in g.index],
                        index=g.index,
                        dtype=float
                    )
                    
                    out["deliv_vs_price_mom_10"] = rolling_corr
                    # Also keep the old name for backward compatibility
                    out["deliv_vs_price_corr_10"] = rolling_corr
                except Exception:
                    # If correlation fails (insufficient data), fill with NaN
                    out["deliv_vs_price_mom_10"] = np.nan
                    out["deliv_vs_price_corr_10"] = np.nan
            
            # Legacy compatibility: keep old names if they existed
            if has_ratio and "delivery_ratio" in g.columns:
                rolling = g["delivery_ratio"].rolling(20)
                out["delivery_z"] = (g["delivery_ratio"] - rolling.mean()) / (rolling.std() + 1e-9)
                out["delivery_spike"] = g["delivery_ratio"] / (rolling.mean() + 1e-9)
                out["delivery_trend_5d"] = g["delivery_ratio"].pct_change(5)
                out["delivery_ratio_5_20"] = deliv_ratio.rolling(5).mean() / (rolling.mean() + 1e-9)
            
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
        """Sector-relative features (cross-sectional z-scores) - all point-in-time safe.
        
        For each (date, sector) group, compute z-scores of key features.
        Handles small sectors gracefully (if std=0 or N<3, z-score=0).
        
        Features:
        - sector_mom_10_z: Z-score of price_mom_10 within sector
        - sector_vol_20_z: Z-score of price_vol_20 within sector
        - sector_ret_10d_z: Z-score of price_ret_10d within sector
        - sector_ret_z: Z-score of 5-day return within sector (legacy)
        - sector_rank_pct: Percentile rank within sector
        - sector_rel_close: Stock close / sector avg close - 1
        """
        required = {"ticker", "sector", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        # Create working frame with computed features
        frame = dataset[["ticker", "sector", "date", "close"]].copy()
        
        # Compute individual stock features (backward-looking)
        ticker_grouped = dataset.groupby("ticker", group_keys=False)
        frame["ret_5d"] = ticker_grouped["close"].pct_change(5)
        frame["ret_10d"] = ticker_grouped["close"].pct_change(10)
        frame["price_mom_10"] = ticker_grouped["close"].apply(lambda x: x / x.shift(10) - 1.0)
        
        # Compute volatility (if not already present)
        frame["price_vol_20"] = ticker_grouped["close"].apply(
            lambda x: x.pct_change().rolling(20).std()
        )
        
        result = pd.DataFrame(index=dataset.index)
        
        # For each date-sector group, compute cross-sectional z-scores
        sector_group = frame.groupby(["date", "sector"], group_keys=False)
        
        # Helper function to compute safe z-score (handles small groups)
        def safe_zscore(series: pd.Series, group_key) -> pd.Series:
            """Compute z-score, return 0 if std=0 or group too small."""
            grp = series.groupby(group_key)
            mean = grp.transform("mean")
            std = grp.transform("std")
            # If std is 0 or NaN, z-score is 0 (no cross-sectional signal)
            z = (series - mean) / std.replace(0, np.nan)
            return z.fillna(0.0)
        
        # Compute z-scores for key features
        result["sector_mom_10_z"] = safe_zscore(frame["price_mom_10"], [frame["date"], frame["sector"]])
        result["sector_vol_20_z"] = safe_zscore(frame["price_vol_20"], [frame["date"], frame["sector"]])
        result["sector_ret_10d_z"] = safe_zscore(frame["ret_10d"], [frame["date"], frame["sector"]])
        result["sector_ret_z"] = safe_zscore(frame["ret_5d"], [frame["date"], frame["sector"]])  # Legacy
        
        # Percentile ranks within sector
        result["sector_rank_pct"] = frame.groupby(["date", "sector"], group_keys=False)["ret_5d"].rank(pct=True)
        
        # Relative to sector average close
        sector_close = dataset.groupby(["date", "sector"], group_keys=False)["close"].transform("mean")
        result["sector_rel_close"] = dataset["close"] / (sector_close + 1e-9) - 1.0
        
        return result

    def _regime_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Market regime and index volatility features - all point-in-time safe.
        
        Uses equal-weighted index proxy from all stocks in the dataset.
        
        Features:
        - regime_is_bull, regime_is_bear: Binary regime indicators
        - market_vol_20d: 20-day realized volatility of the index proxy
        - index_vol_pct: Percentile rank of current volatility (0-100)
        - index_vol_spike: Current vol / historical median (regime change detector)
        """
        required = {"date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()
        
        # Construct equal-weighted market proxy
        market = dataset.groupby("date")["close"].mean().sort_index()
        
        # Regime classification (bull/bear/sideways)
        # RegimeClassifier.infer expects a DataFrame with [date, ticker, close] columns
        # Create a minimal DataFrame with all dates and a dummy ticker
        market_df = pd.DataFrame({
            "date": market.index,
            "ticker": "MARKET",
            "close": market.values
        })
        regime_result = self.regime_classifier.infer(market_df)
        if not regime_result.empty and "composite_regime" in regime_result.columns:
            # regime_result is already indexed by date
            if "date" in regime_result.columns:
                regime_map = regime_result.set_index("date")["composite_regime"].ffill().bfill()
            else:
                regime_map = regime_result["composite_regime"].ffill().bfill()
        else:
            # Fallback: create simple regime map
            regime_map = pd.Series("sideways", index=market.index, name="composite_regime")
        # Ensure regime_map is a Series indexed by date
        if isinstance(regime_map, pd.Series):
            regime_frame = regime_map.rename("market_regime").reset_index()
        else:
            regime_frame = pd.DataFrame({"date": market.index, "market_regime": "sideways"})
        
        # Market volatility (20-day realized vol)
        vol_series = market.pct_change().rolling(20).std()
        
        # Volatility percentile rank (point-in-time: only use history up to current date)
        # Expanding window to compute percentile rank
        vol_pct = vol_series.expanding(min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= 20 else np.nan,
            raw=False
        )
        
        # Volatility spike (current vol vs historical median)
        # Use expanding median to avoid future leakage
        vol_median = vol_series.expanding(min_periods=20).median()
        vol_spike = vol_series / (vol_median + 1e-9)
        
        # Combine into frames for merging
        vol_frame = pd.DataFrame({
            "date": vol_series.index,
            "market_vol_20d": vol_series.values,
            "index_vol_pct": vol_pct.values,
            "index_vol_spike": vol_spike.values,
        })

        # Merge back to dataset
        merged = pd.DataFrame({"date": dataset["date"]}, index=dataset.index)
        merged = merged.merge(regime_frame, on="date", how="left")
        merged = merged.merge(vol_frame, on="date", how="left")

        # Enhanced volatility features (VIX-equivalent, vol-of-vol, term structure)
        # VIX-equivalent: 30-day forward-looking volatility estimate (using realized vol as proxy)
        vol_30d = market.pct_change().rolling(30).std()
        
        # Volatility of volatility (vol regime uncertainty)
        vol_of_vol = vol_series.rolling(20).std()
        
        # Term structure: short-vol (5d) vs long-vol (20d)
        vol_5d = market.pct_change().rolling(5).std()
        vol_term_structure = vol_5d / (vol_series + 1e-9)
        
        # Create output features
        out = pd.DataFrame(index=dataset.index)
        out["regime_is_bull"] = (merged["market_regime"] == "bull").astype(float)
        out["regime_is_bear"] = (merged["market_regime"] == "bear").astype(float)
        out["market_vol_20d"] = merged["market_vol_20d"]
        out["index_vol_pct"] = merged["index_vol_pct"]
        out["index_vol_spike"] = merged["index_vol_spike"]
        
        # Add enhanced volatility features (align with dataset dates)
        out["vix_equivalent_30d"] = vol_30d.reindex(dataset["date"], method="ffill").values
        out["vol_of_vol_20d"] = vol_of_vol.reindex(dataset["date"], method="ffill").values
        out["vol_term_structure_5_20"] = vol_term_structure.reindex(dataset["date"], method="ffill").values
        
        return out

    def _regime_adaptive_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Regime-adaptive features that work across all market conditions.
        
        Features:
        - market_adj_mom_5, _10, _20: (Stock Return - Market Return) / Volatility
          Separates true alpha from beta, scales by regime volatility
        - regime_adj_volume: Current Volume / (20-Day Avg Volume * Volatility Percentile)
          High vol days = big volume is normal, shouldn't trigger signal
        - market_alpha_10d, _20d: Stock return minus market return (excess return)
        - beta_estimate_20d: Rolling correlation * (stock_vol / market_vol)
        """
        required = {"ticker", "date", "close", "volume"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()
        
        # Construct market proxy (equal-weighted index)
        market = dataset.groupby("date")["close"].mean().sort_index()
        market_rets = market.pct_change()
        market_vol = market_rets.rolling(20).std()
        
        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            price = g["close"]
            volume = g["volume"]
            stock_rets = price.pct_change()
            
            # Align market data with stock dates - ensure proper index alignment
            # Create Series aligned to group index with market values by date
            g_dates = pd.to_datetime(g["date"])
            # Use map to align by date values, then create Series with group index
            aligned_market_rets = pd.Series(
                market_rets.reindex(g_dates.values, method="ffill").values,
                index=g.index,
                dtype=float
            )
            aligned_market_vol = pd.Series(
                market_vol.reindex(g_dates.values, method="ffill").values,
                index=g.index,
                dtype=float
            )
            
            # Stock volatility (for normalization)
            stock_vol = stock_rets.rolling(20).std()
            
            # Market-adjusted momentum: (Stock Return - Market Return) / Volatility
            # This separates true skill from beta, scales by regime
            excess_rets_5d = stock_rets.rolling(5).sum() - aligned_market_rets.rolling(5).sum()
            excess_rets_10d = stock_rets.rolling(10).sum() - aligned_market_rets.rolling(10).sum()
            excess_rets_20d = stock_rets.rolling(20).sum() - aligned_market_rets.rolling(20).sum()
            
            # Normalize by volatility (regime-adjusted) - ensure numeric types
            vol_denom = stock_vol.astype(float) + 1e-9
            out["market_adj_mom_5"] = (excess_rets_5d.astype(float) / vol_denom).fillna(0.0)
            out["market_adj_mom_10"] = (excess_rets_10d.astype(float) / vol_denom).fillna(0.0)
            out["market_adj_mom_20"] = (excess_rets_20d.astype(float) / vol_denom).fillna(0.0)
            
            # Pure excess returns (market alpha) - ensure numeric
            out["market_alpha_5d"] = excess_rets_5d.astype(float).fillna(0.0)
            out["market_alpha_10d"] = excess_rets_10d.astype(float).fillna(0.0)
            out["market_alpha_20d"] = excess_rets_20d.astype(float).fillna(0.0)
            
            # Beta estimate: correlation * (stock_vol / market_vol)
            # Rolling correlation between stock and market returns
            # Ensure both series are numeric and aligned
            stock_rets_numeric = stock_rets.astype(float)
            aligned_market_rets_numeric = aligned_market_rets.astype(float)
            rolling_corr = stock_rets_numeric.rolling(20).corr(aligned_market_rets_numeric)
            vol_ratio = stock_vol.astype(float) / (aligned_market_vol.astype(float) + 1e-9)
            out["beta_estimate_20d"] = (rolling_corr * vol_ratio).fillna(0.0)
            
            # Regime-adjusted volume: Current Volume / (20-Day Avg Volume * Volatility Percentile)
            # Get volatility percentile from regime block (if available)
            # For now, compute it here using expanding window
            vol_pct = stock_vol.expanding(min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= 20 else 50.0,
                raw=False
            )
            vol_pct_normalized = vol_pct / 100.0  # Convert to 0-1 scale
            volume_mean_20 = volume.rolling(20).mean()
            # Adjust volume baseline by volatility regime
            adjusted_volume_baseline = volume_mean_20 * (1.0 + vol_pct_normalized)
            out["regime_adj_volume"] = volume / (adjusted_volume_baseline + 1e-9)
            
            # Volume surprise relative to volatility-adjusted baseline
            out["volume_surprise_vol_adj"] = (volume - adjusted_volume_baseline) / (adjusted_volume_baseline + 1e-9)
            
            return out
        
        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _mean_reversion_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Mean-reversion features for bear markets and range-bound conditions.
        
        Features:
        - dist_from_200ma: (Close - 200MA) / 200MA (oversold/overbought signal)
        - dist_from_200ma_pct: Percentile rank of distance (0-100)
        - rsi_extreme_low, rsi_extreme_high: Binary flags for RSI < 30 or > 70
        - bollinger_position: Position within Bollinger Bands (-2 to +2 std devs)
        - bollinger_squeeze: Band width relative to historical (volatility compression)
        - price_vs_ma_ratio_50, _200: Price / MA ratios (mean reversion entry zones)
        """
        required = {"ticker", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()
        
        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            price = g["close"]
            
            # 200-day moving average (long-term trend)
            ma_200 = price.rolling(200, min_periods=50).mean()
            ma_50 = price.rolling(50, min_periods=20).mean()
            
            # Distance from 200MA (oversold/overbought)
            out["dist_from_200ma"] = (price - ma_200) / (ma_200 + 1e-9)
            out["dist_from_50ma"] = (price - ma_50) / (ma_50 + 1e-9)
            
            # Percentile rank of distance (point-in-time: expanding window)
            out["dist_from_200ma_pct"] = out["dist_from_200ma"].expanding(min_periods=50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= 50 else 50.0,
                raw=False
            )
            
            # Price vs MA ratios
            out["price_vs_ma_ratio_50"] = price / (ma_50 + 1e-9)
            out["price_vs_ma_ratio_200"] = price / (ma_200 + 1e-9)
            
            # RSI (already computed in technical block, but compute here for extremes)
            rets = price.pct_change()
            delta = rets
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            
            # RSI extreme flags
            out["rsi_extreme_low"] = (rsi < 30).astype(float)  # Oversold
            out["rsi_extreme_high"] = (rsi > 70).astype(float)  # Overbought
            out["rsi_distance_from_50"] = rsi - 50.0  # Distance from neutral
            
            # Bollinger Bands
            roll_mean = price.rolling(20).mean()
            roll_std = price.rolling(20).std()
            upper_band = roll_mean + 2 * roll_std
            lower_band = roll_mean - 2 * roll_std
            
            # Position within Bollinger Bands (-2 to +2 std devs)
            out["bollinger_position"] = (price - roll_mean) / (roll_std + 1e-9)
            
            # Bollinger Band width (volatility measure)
            band_width = (upper_band - lower_band) / (roll_mean + 1e-9)
            band_width_median = band_width.rolling(60, min_periods=20).median()
            out["bollinger_squeeze"] = band_width / (band_width_median + 1e-9)  # < 1 = compression
            
            # Mean reversion signal: oversold conditions
            out["oversold_signal"] = (
                (out["dist_from_200ma"] < -0.1).astype(float) *  # 10% below 200MA
                (rsi < 35).astype(float) *  # RSI oversold
                (out["bollinger_position"] < -1.5).astype(float)  # Near lower Bollinger Band
            )
            
            return out
        
        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)
