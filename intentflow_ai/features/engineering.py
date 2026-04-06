"""Feature layer composition for IntentFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from intentflow_ai.config.settings import settings
from intentflow_ai.data.universe import load_universe
from intentflow_ai.modeling.regimes import RegimeClassifier
from intentflow_ai.utils.logging import get_logger
from intentflow_ai.features.modern_features import build_modern_features
from sklearn.linear_model import LinearRegression

logger = get_logger(__name__)


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
                "momentum_enhanced": self._momentum_enhanced_block,  # NEW: From Qlib
                "volatility": self._volatility_block,
                "atr": self._atr_block,
                "turnover": self._turnover_block,
                # "ownership": self._ownership_block,  # DISABLED v5: data corrupt, always returns empty
                "delivery": self._delivery_block,
                # "fundamental": self._fundamental_block,  # DISABLED: +0.0025 IC without it (ablation study Dec 2024)
                # "narrative": self._narrative_block,  # DISABLED v5: no sentiment data, always returns empty
                "sector_relative": self._sector_relative_block,
                # "regime": self._regime_block,  # DISABLED: Causing negative IC (volatility bias)
                # "regime_adaptive": self._regime_adaptive_block, # DISABLED: Causing negative IC
                "mean_reversion": self._mean_reversion_block,
                "mean_reversion_enhanced": self._mean_reversion_enhanced_block,  # NEW: From Qlib
                "volume_enhanced": self._volume_enhanced_block,  # NEW: From Qlib
                "ranking": self._ranking_block,  # NEW: Cross-sectional ranks
                "orthogonal": self._orthogonal_block,
                "sector_momentum": self._sector_momentum_block,  # NEW: Sector relative performance
                "earnings_metrics": self._earnings_metrics_block,  # NEW: Earnings surprise, growth
                "quality_scores": self._quality_scores_block,  # NEW: ROE, margins, cash conversion
                # "financial_ratios": self._financial_ratios_block,  # DISABLED v5: always returns empty, overlaps earnings_metrics
                "sector_normalized": self._sector_normalized_block,  # NEW: Sector-relative features
                # "seasonality": self._seasonality_block,  # DISABLED: IC drop 2024, likely overfitting to calendar effects (ablation Dec 2024)
                "macro": self._macro_block,  # NEW: Macro features (VIX, USD/INR, Crude, FII/DII)
                "options": self._options_block,  # NEW: Options sentiment (PCR, Max Pain)
                # "modern_market": self._modern_market_block,  # DISABLED: +0.0078 IC without it (ablation study Dec 2024)
                "fii_dii_flow": self._fii_dii_flow_block,  # FII/DII institutional flow features
                "global_overnight": self._global_overnight_block,  # V5: Global market overnight returns
                "delivery_momentum": self._delivery_momentum_block,  # V5: Delivery × momentum interactions
            }

    @staticmethod
    def _group_apply(grouped: pd.core.groupby.DataFrameGroupBy, func) -> pd.DataFrame:
        try:
            return grouped.apply(func, include_groups=False)
        except TypeError:
            return grouped.apply(func)

    def build(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # === WATERFALL LOGGING: Track ticker count at each step ===
        initial_tickers = dataset['ticker'].nunique() if 'ticker' in dataset.columns else 0
        logger.info(f"[FeatureEngineer.build] STEP 0: Input dataset has {initial_tickers} unique tickers, {len(dataset)} rows")
        
        dataset = dataset.copy()
        if "date" in dataset.columns:
            dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        if "sector" not in dataset.columns or dataset["sector"].isna().any():
            lookup = _sector_lookup()
            if not lookup.empty and "ticker" in dataset.columns:
                dataset["sector"] = dataset.get("sector", pd.Series(index=dataset.index, dtype="string"))
                dataset["sector"] = dataset["sector"].fillna(dataset["ticker"].map(lookup))
        
        logger.info(f"[FeatureEngineer.build] STEP 1: After date/sector prep, {dataset['ticker'].nunique() if 'ticker' in dataset.columns else 0} tickers, {len(dataset)} rows")
        
        frames: List[pd.DataFrame] = []
        for name, builder in self.feature_blocks.items():
            block = builder(dataset.copy())
            tickers_before = dataset['ticker'].nunique() if 'ticker' in dataset.columns else 0
            if block.empty:
                logger.warning(f"[FeatureEngineer.build] Feature block '{name}' returned EMPTY, skipping")
                continue
            block.columns = [f"{name}__{col}" for col in block.columns]
            frames.append(block)
            logger.info(f"[FeatureEngineer.build] STEP 2.{len(frames)}: After '{name}' block, added {len(block.columns)} features ({len(frames)} blocks total)")
        
        if frames:
            combined = pd.concat(frames, axis=1)
            logger.info(f"[FeatureEngineer.build] STEP 3: After concat all blocks, {len(combined.columns)} total features, {len(combined)} rows")
            
            combined = combined.apply(pd.to_numeric, errors="coerce")
            logger.info(f"[FeatureEngineer.build] STEP 4: After to_numeric coercion, {len(combined)} rows")
            
            combined = combined.replace([np.inf, -np.inf], np.nan)
            logger.info(f"[FeatureEngineer.build] STEP 5: After replacing inf with NaN, {len(combined)} rows")
            
            # === PRIORITY 2: WINSORIZATION (1%-99%) ===
            # Clip extreme values to reduce outlier impact on z-scores
            for col in combined.columns:
                lower = combined[col].quantile(0.01)
                upper = combined[col].quantile(0.99)
                combined[col] = combined[col].clip(lower=lower, upper=upper)
            logger.info(f"[FeatureEngineer.build] STEP 6: After winsorization (1%-99%), {len(combined)} rows")
            
            # NOTE: Removed universe-wide z-score per date (caused double normalization)
            # Sector z-scores are already applied in individual feature blocks (_sector_relative)
            
            # === PRIORITY 3: FACTOR INTERACTION FEATURES ===
            # Add interaction terms between key factors
            mom_cols = [c for c in combined.columns if "momentum" in c.lower() or "mom_" in c.lower() or "ret_" in c.lower()]
            val_cols = [c for c in combined.columns if "pe_" in c.lower() or "pb_" in c.lower() or "value" in c.lower()]
            qual_cols = [c for c in combined.columns if "roe" in c.lower() or "quality" in c.lower() or "margin" in c.lower()]
            
            # Pick representative features for interaction
            mom_rep = next((c for c in mom_cols if "10" in c), mom_cols[0] if mom_cols else None)
            val_rep = next((c for c in val_cols if "sector_z" in c), val_cols[0] if val_cols else None)
            qual_rep = next((c for c in qual_cols if "sector_z" in c), qual_cols[0] if qual_cols else None)
            
            if mom_rep and val_rep:
                combined["interaction__value_momentum"] = combined[val_rep] * combined[mom_rep]
            if mom_rep and qual_rep:
                combined["interaction__quality_momentum"] = combined[qual_rep] * combined[mom_rep]
            if val_rep and qual_rep:
                combined["interaction__value_quality"] = combined[val_rep] * combined[qual_rep]
            logger.info(f"[FeatureEngineer.build] STEP 7: After factor interactions, {len(combined.columns)} total features")
            
            result = combined.fillna(0.0)
            logger.info(f"[FeatureEngineer.build] STEP 8: After fillna(0.0), {len(result)} rows (FINAL)")
            return result
        
        logger.warning(f"[FeatureEngineer.build] No feature blocks succeeded, using baseline features")
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
        """
        Fundamental features using Screener.in for NSE stocks.
        
        Fetches and computes:
        - Valuation: P/E sector-relative, value composite
        - Profitability: ROE, ROCE sector-relative  
        - Quality: Operating profit metrics
        """
        # Check if we have required columns
        if "ticker" not in dataset.columns or "date" not in dataset.columns:
            return pd.DataFrame()
        
        try:
            from intentflow_ai.data.fundamentals_provider import get_fundamental_provider
            from intentflow_ai.features.fundamental_features import FundamentalFeatures
            
            # Try loading from EODHD parquet first (Professional Data Lake)
            eodhd_path = Path(settings.data_dir) / "processed" / "fundamentals_eodhd.parquet"
            poc_path = Path(settings.data_dir) / "processed" / "fundamentals_poc.parquet"
            csv_path = Path(settings.data_dir) / "fundamentals.csv"
            
            if eodhd_path.exists():
                fundamentals = pd.read_parquet(eodhd_path)
                print(f"✅ Loaded {len(fundamentals)} fundamental records from EODHD parquet")
                # Ensure date columns are datetime
                for col in ['date', 'report_date', 'available_date']:
                    if col in fundamentals.columns:
                        fundamentals[col] = pd.to_datetime(fundamentals[col])
            elif poc_path.exists():
                fundamentals = pd.read_parquet(poc_path)
                print(f"✅ Loaded {len(fundamentals)} fundamental records from POC parquet")
                # Ensure date columns are datetime
                for col in ['date', 'report_date', 'available_date']:
                    if col in fundamentals.columns:
                        fundamentals[col] = pd.to_datetime(fundamentals[col])
            elif csv_path.exists():
                fundamentals = pd.read_csv(csv_path)
                # Ensure date columns are datetime
                if 'date' in fundamentals.columns:
                    fundamentals['date'] = pd.to_datetime(fundamentals['date'])
                if 'report_date' in fundamentals.columns:
                    fundamentals['report_date'] = pd.to_datetime(fundamentals['report_date'])
                if 'available_date' in fundamentals.columns:
                    fundamentals['available_date'] = pd.to_datetime(fundamentals['available_date'])
            else:
                # Fallback to fetching (slow/rate-limited)
                # Get unique symbols and date range
                symbols = dataset['ticker'].unique().tolist()
                start_date = dataset['date'].min()
                end_date = dataset['date'].max()
                
                # Fetch fundamentals
                provider = get_fundamental_provider()
                all_fundamentals = []
                
                for symbol in symbols:
                    fund_df = provider.fetch_fundamentals(symbol, start_date, end_date)
                    if not fund_df.empty:
                        all_fundamentals.append(fund_df)
                
                if not all_fundamentals:
                    # No fundamental data available
                    return pd.DataFrame()
                
                fundamentals = pd.concat(all_fundamentals, ignore_index=True)
            
            # Compute fundamental features
            feature_engine = FundamentalFeatures()
            features = feature_engine.compute_all_features(dataset, fundamentals)
            
            # Prefix with "fundamental__" is already done in compute_all_features
            return features
            
        except Exception as e:
            print(f"Warning: Failed to compute fundamental features: {e}")
            return pd.DataFrame()

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
        
        # Construct equal-weighted market proxy (expanding window to prevent leakage)
        # We use expanding mean to simulate "what we knew at time t"
        market = dataset.groupby("date")["close"].mean().sort_index()
        # For regime detection, we need a stable history, so we'll use the expanding mean
        # of the daily average close prices.
        # Note: Ideally we'd use an index ticker, but this is a good proxy.

        
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
        # Use expanding window to prevent future leakage
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

    def _orthogonal_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Orthogonalize key features against market beta.
        
        Computes residuals of features regressed on market returns/volatility.
        Features:
        - idio_ret_10d: 10-day return orthogonal to market return
        - idio_mom_10: Momentum orthogonal to market momentum
        - idio_vol_20: Volatility orthogonal to market volatility
        """
        required = {"ticker", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        # 1. Compute Market Factors (Point-in-Time Safe)
        # Use expanding window for market mean to avoid future leakage
        market = dataset.groupby("date")["close"].mean().sort_index()
        market_ret = market.pct_change()
        market_ret_10d = market.pct_change(10)
        market_vol = market_ret.rolling(20).std()
        
        # Align market factors to dataset
        # Create a DataFrame with market factors indexed by date
        market_factors = pd.DataFrame({
            "mkt_ret": market_ret,
            "mkt_ret_10d": market_ret_10d,
            "mkt_vol": market_vol
        })
        
        # Merge market factors into dataset for easy regression
        # Reset index to preserve original index after merge
        df_work = dataset[["date", "ticker", "close"]].copy()
        df_work["ret_1d"] = df_work.groupby("ticker")["close"].pct_change()
        df_work["ret_10d"] = df_work.groupby("ticker")["close"].pct_change(10)
        df_work["vol_20"] = df_work.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().rolling(20).std()
        )
        
        df_work = df_work.merge(market_factors, on="date", how="left")
        
        # 2. Compute Residuals (Vectorized per group is hard, so we iterate or use simple beta approx)
        # For speed/simplicity in this phase, we'll use a rolling beta approximation
        # Residual = Stock_Feat - Beta * Market_Feat
        
        def compute_residuals(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            
            # Idiosyncratic Return (10d)
            # Beta = Cov(Stock, Mkt) / Var(Mkt)
            # We use a rolling 60-day window for beta
            cov_10d = g["ret_10d"].rolling(60).cov(g["mkt_ret_10d"])
            var_10d = g["mkt_ret_10d"].rolling(60).var()
            beta_10d = (cov_10d / (var_10d + 1e-9)).fillna(1.0)
            out["idio_ret_10d"] = g["ret_10d"] - (beta_10d * g["mkt_ret_10d"])
            
            # Idiosyncratic Volatility
            # Simple difference for now (Vol Spread) or orthogonal
            # Regressing Vol against Vol
            cov_vol = g["vol_20"].rolling(60).cov(g["mkt_vol"])
            var_vol = g["mkt_vol"].rolling(60).var()
            beta_vol = (cov_vol / (var_vol + 1e-9)).fillna(1.0)
            out["idio_vol_20"] = g["vol_20"] - (beta_vol * g["mkt_vol"])
            
            return out

        return self._group_apply(df_work.groupby("ticker", group_keys=False), compute_residuals)

    def _momentum_enhanced_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Enhanced momentum features from Qlib Alpha158.
        
        Features:
        - ma5_ratio, ma20_ratio, ma60_ratio: Close vs moving averages
        - mom_accel: Momentum acceleration (short-term vs long-term momentum change)
        - momentum_quality: Momentum with volume confirmation
        """
        required = {"ticker", "date", "close", "volume"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            close = g["close"]
            volume = g["volume"]
            
            # MA ratios (Qlib feature)
            ma5 = close.rolling(5).mean()
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            
            out["ma5_ratio"] = (close / ma5 - 1.0).fillna(0)
            out["ma20_ratio"] = (close / ma20 - 1.0).fillna(0)
            out["ma60_ratio"] = (close / ma60 - 1.0).fillna(0)
            
            # Momentum acceleration (change in momentum)
            ret_10d = close.pct_change(10)
            ret_20d = close.pct_change(20)
            out["mom_accel"] = (ret_10d - ret_20d).fillna(0)
            
            # Momentum quality: momentum with above-average volume
            vol_avg = volume.rolling(20).mean()
            vol_high = (volume > vol_avg).astype(int)
            out["mom_quality"] = (ret_10d * vol_high).fillna(0)
            
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _mean_reversion_enhanced_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Enhanced mean-reversion features from Qlib Alpha158 and MLAT.
        
        Features:
        - dist_to_max_20d, dist_to_min_20d: Distance from recent extremes
        - recent_extreme_ratio: How close to max vs min
        - range_position_20d: Position within recent range (0-1)
        """
        required = {"ticker", "date", "close"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            close = g["close"]
            
            # Distance from recent max/min (Qlib feature)
            max_20d = close.rolling(20).max()
            min_20d = close.rolling(20).min()
            
            out["dist_to_max_20d"] = (close / max_20d - 1.0).fillna(0)
            out["dist_to_min_20d"] = (close / min_20d - 1.0).fillna(0)
            
            # Position within range (0 = at min, 1 = at max)
            range_20d = max_20d - min_20d
            out["range_position_20d"] = ((close - min_20d) / (range_20d + 1e-9)).fillna(0.5)
            
            # Extreme ratio: are we closer to max or min?
            dist_to_max = max_20d - close
            dist_to_min = close - min_20d
            out["recent_extreme_ratio"] = ((dist_to_min - dist_to_max) / (range_20d + 1e-9)).fillna(0)
            
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _volume_enhanced_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Enhanced volume features from Qlib Alpha158.
        
        Features:
        - pv_corr_10d, pv_corr_20d: Price-volume correlation
        - vol_trend: Volume momentum (short vs long MA)
        - vol_spike: Volume shock detection
        """
        required = {"ticker", "date", "close", "volume"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        def compute(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            close = g["close"]
            volume = g["volume"]
            
            # Price-volume correlation (Qlib feature)
            returns = close.pct_change()
            vol_changes = volume.pct_change()
            
            out["pv_corr_10d"] = returns.rolling(10).corr(vol_changes).fillna(0)
            out["pv_corr_20d"] = returns.rolling(20).corr(vol_changes).fillna(0)
            
            # Volume trend (momentum in volume)
            vol_ma5 = volume.rolling(5).mean()
            vol_ma20 = volume.rolling(20).mean()
            out["vol_trend"] = ((vol_ma5 / (vol_ma20 + 1e-9)) - 1.0).fillna(0)
            
            # Volume spike detection
            vol_mean = volume.rolling(20).mean()
            vol_std = volume.rolling(20).std()
            out["vol_spike"] = ((volume - vol_mean) / (vol_std + 1e-9)).fillna(0)
            
            return out

        return self._group_apply(dataset.groupby("ticker", group_keys=False), compute)

    def _ranking_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional ranking features (critical for long-short strategies).
        
        Features:
        - ret_20d_rank, vol_20_rank, volume_rank: Percentile ranks by date
        
        These are essential for relative value strategies.
        """
        required = {"ticker", "date", "close", "volume"}
        if not required.issubset(dataset.columns):
            return pd.DataFrame()

        # First compute the base features we'll rank
        def compute_base(group: pd.DataFrame) -> pd.DataFrame:
            g = group.sort_values("date")
            out = pd.DataFrame(index=g.index)
            close = g["close"]
            volume = g["volume"]
            
            # Features to rank
            out["ret_20d_raw"] = close.pct_change(20)
            out["vol_20_raw"] = close.pct_change().rolling(20).std()
            out["volume_raw"] = volume.rolling(5).mean()  # Smoothed volume
            
            return out

        # Compute base features
        base_features = self._group_apply(dataset.groupby("ticker", group_keys=False), compute_base)
        temp_df = dataset[["ticker", "date"]].copy()
        temp_df = temp_df.join(base_features)
        
        # Compute cross-sectional ranks within each date
        result = pd.DataFrame(index=dataset.index)
        result["ret_20d_rank"] = temp_df.groupby("date")["ret_20d_raw"].rank(pct=True).fillna(0.5)
        result["vol_20_rank"] = temp_df.groupby("date")["vol_20_raw"].rank(pct=True).fillna(0.5)
        result["volume_rank"] = temp_df.groupby("date")["volume_raw"].rank(pct=True).fillna(0.5)
        
        return result

    def _sector_momentum_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        try:
            from intentflow_ai.features.advanced_features import sector_momentum_features
            
            return sector_momentum_features(dataset)
        except Exception as e:
            logger.warning(f"Sector momentum feature computation failed: {e}")
            return pd.DataFrame(index=dataset.index)
    
    def _earnings_metrics_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute earnings features from screener.in quarterly data merged into panel."""
        out = pd.DataFrame(index=dataset.index)
        try:
            # These columns come from load_enhanced_panel() → screener merge
            if "eps" not in dataset.columns or dataset["eps"].notna().sum() < 10:
                return out

            g = dataset.groupby("ticker", group_keys=False)

            if "eps" in dataset.columns:
                out["earnings__eps"] = dataset["eps"]
                out["earnings__eps_growth_qoq"] = g["eps"].pct_change(1, fill_method=None)

            if "revenue" in dataset.columns:
                out["earnings__revenue"] = dataset["revenue"]
                out["earnings__revenue_growth_qoq"] = g["revenue"].pct_change(1, fill_method=None)

            if "net_income" in dataset.columns and "revenue" in dataset.columns:
                rev = dataset["revenue"].replace(0, float("nan"))
                out["earnings__net_margin"] = dataset["net_income"] / rev

            if "operating_profit" in dataset.columns and "revenue" in dataset.columns:
                rev = dataset["revenue"].replace(0, float("nan"))
                out["earnings__operating_margin"] = dataset["operating_profit"] / rev

            if "pe_ratio" in dataset.columns:
                out["earnings__pe_ratio"] = dataset["pe_ratio"]

            if "roe" in dataset.columns:
                out["earnings__roe"] = dataset["roe"]

            if "roce" in dataset.columns:
                out["earnings__roce"] = dataset["roce"]

            if "dividend_yield" in dataset.columns:
                out["earnings__dividend_yield"] = dataset["dividend_yield"]

            if "book_value_per_share" in dataset.columns:
                out["earnings__book_value"] = dataset["book_value_per_share"]

            # Drop columns that are all NaN
            out = out.dropna(axis=1, how="all")
            return out

        except Exception as e:
            logger.warning(f"Earnings metrics feature computation failed: {e}")
            return pd.DataFrame(index=dataset.index)

    def _quality_scores_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Quality scores derived from fundamentals."""
        out = pd.DataFrame(index=dataset.index)
        try:
            has_margin = "operating_profit" in dataset.columns and "revenue" in dataset.columns
            has_roe = "roe" in dataset.columns

            if not has_margin and not has_roe:
                return out

            if has_margin:
                rev = dataset["revenue"].replace(0, float("nan"))
                margin = dataset["operating_profit"] / rev
                out["quality__op_margin"] = margin

            if has_roe:
                out["quality__roe"] = dataset["roe"]

            out = out.dropna(axis=1, how="all")
            return out
        except Exception as e:
            logger.warning(f"Quality scores feature computation failed: {e}")
            return pd.DataFrame(index=dataset.index)

    def _financial_ratios_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Financial ratios from fundamental data."""
        out = pd.DataFrame(index=dataset.index)
        try:
            if "pe_ratio" in dataset.columns and dataset["pe_ratio"].notna().sum() > 10:
                out["ratio__pe"] = dataset["pe_ratio"]
                # Sector-relative PE
                if "sector" in dataset.columns:
                    out["ratio__pe_sector_z"] = dataset.groupby("sector", group_keys=False).apply(
                        lambda g: (g["pe_ratio"] - g["pe_ratio"].mean()) / g["pe_ratio"].std().clip(lower=0.1)
                    )

            if "book_value_per_share" in dataset.columns and "close" in dataset.columns:
                bv = dataset["book_value_per_share"].replace(0, float("nan"))
                out["ratio__pb"] = dataset["close"] / bv

            out = out.dropna(axis=1, how="all")
            return out
        except Exception as e:
            logger.warning(f"Financial ratios computation failed: {e}")
            return pd.DataFrame(index=dataset.index)
    
    def _sector_normalized_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        try:
            from intentflow_ai.features.financial_ratios import sector_normalized_features
            
            return sector_normalized_features(dataset)
        except Exception as e:
            logger.warning(f"Sector normalization failed: {e}")
            return pd.DataFrame(index=dataset.index)
    
    def _seasonality_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add Indian market seasonality features.
        
        Features include:
        - Diwali period (October-November)
        - Budget season (January-February)
        - Earnings seasons (quarterly)
        - F&O expiry effects (monthly)
        - Sector-specific seasonality
        """
        try:
            from intentflow_ai.features.seasonality import add_seasonality_to_df
            
            # Ensure date column exists
            if "date" not in dataset.columns:
                logger.warning("Seasonality block: 'date' column missing, skipping")
                return pd.DataFrame(index=dataset.index)
            
            # Filter out invalid dates before processing
            dataset_clean = dataset.copy()
            dataset_clean["date"] = pd.to_datetime(dataset_clean["date"], errors="coerce")
            valid_dates = dataset_clean["date"].notna()
            
            if not valid_dates.any():
                logger.warning("Seasonality block: No valid dates found, skipping")
                return pd.DataFrame(index=dataset.index)
            
            # Get sector column (may not exist)
            sector_col = "sector" if "sector" in dataset_clean.columns else None
            
            # Add seasonality features only on valid dates
            try:
                result_df = add_seasonality_to_df(
                    dataset_clean[valid_dates],
                    date_col="date",
                    sector_col=sector_col,
                )
            except Exception as inner_e:
                logger.warning(f"Seasonality computation error: {inner_e}, skipping")
                return pd.DataFrame(index=dataset.index)
            
            # Extract only the seasonality feature columns (exclude date/sector)
            seasonal_cols = [
                col for col in result_df.columns
                if col not in ["date", "sector"] and col.startswith((
                    "is_", "days_", "month", "quarter", "day_of", "week_of",
                    "sector_seasonality"
                ))
            ]
            
            if seasonal_cols:
                # Reindex to match original dataset index
                result = pd.DataFrame(index=dataset.index)
                for col in seasonal_cols:
                    result.loc[valid_dates, col] = result_df[col].values
                result = result.fillna(0.0)  # Fill invalid dates with 0
                return result
            else:
                logger.warning("Seasonality block: No seasonality features generated")
                return pd.DataFrame(index=dataset.index)
                
        except Exception as e:
            logger.warning(f"Seasonality feature computation failed: {e}", exc_info=True)
            return pd.DataFrame(index=dataset.index)
    
    def _macro_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add macro-economic features.
        
        Features include:
        - India VIX (fear gauge)
        - USD/INR exchange rate
        - Crude oil prices
        - US 10Y Treasury yield
        - FII/DII flows
        - NIFTY regime indicators
        """
        try:
            from intentflow_ai.data.macro_provider import MacroDataProvider
            
            # Ensure date column exists
            if "date" not in dataset.columns:
                logger.warning("Macro block: 'date' column missing, skipping")
                return pd.DataFrame(index=dataset.index)
            
            # Get unique dates from dataset
            dates = pd.to_datetime(dataset["date"].unique())
            if len(dates) == 0:
                logger.warning("Macro block: No valid dates found, skipping")
                return pd.DataFrame(index=dataset.index)
            
            # Initialize macro provider
            provider = MacroDataProvider()
            
            # Get macro data for date range
            start_date = dates.min()
            end_date = dates.max()
            macro_df = provider.get_macro_df(start_date, end_date)
            
            if macro_df.empty:
                logger.warning("Macro block: No macro data retrieved, returning empty features")
                return pd.DataFrame(index=dataset.index)
            
            # Merge macro features back to dataset by date
            dataset_dates = pd.to_datetime(dataset["date"])
            result = pd.DataFrame(index=dataset.index)
            
            # For each macro feature column, map to dataset dates
            for col in macro_df.columns:
                # Create a series mapping dates to macro values
                macro_series = macro_df[col].reindex(dataset_dates, method="ffill")
                result[col] = macro_series.values
            
            # Fill NaN values with 0 (for dates before macro data availability)
            result = result.fillna(0.0)
            
            logger.info(f"Macro block: Added {len(result.columns)} macro features")
            return result
            
        except Exception as e:
            logger.warning(f"Macro feature computation failed: {e}", exc_info=True)
            return pd.DataFrame(index=dataset.index)

    def _options_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add options-based sentiment features.
        
        Features:
        - NIFTY PCR (Put-Call Ratio)
        - PCR Z-Score (Sentiment extreme)
        - F&O Stock Flag
        - Options Sentiment Composite
        """
        try:
            from intentflow_ai.features.options_features import add_options_features_to_df
            from intentflow_ai.data.nse_options_provider import get_nse_options_provider
            
            # Ensure required columns exist
            if "ticker" not in dataset.columns:
                return pd.DataFrame(index=dataset.index)
                
            # Get options provider
            provider = get_nse_options_provider()
            
            # Fetch market-wide PCR (NIFTY)
            # Note: Currently fetches latest/stubbed data. 
            # For strict backtesting, this needs historical point-in-time data.
            # The provider handles caching and graceful failure (returns NaNs).
            nifty_data = provider.get_pcr("NIFTY")
            nifty_pcr = nifty_data.get("pcr", np.nan)
            
            # Apply features
            # This adds: nifty_pcr, nifty_pcr_zscore, nifty_pcr_sentiment, is_fno_stock
            result_df = add_options_features_to_df(
                dataset,
                ticker_col="ticker",
                close_col="close" if "close" in dataset.columns else None,
                nifty_pcr=nifty_pcr
            )
            
            # Extract only the new options columns
            # We identify them by checking what wasn't in the original dataset
            new_cols = [c for c in result_df.columns if c not in dataset.columns]
            
            if not new_cols:
                return pd.DataFrame(index=dataset.index)
                
            return result_df[new_cols].fillna(0.0)
            
        except Exception as e:
            logger.warning(f"Options feature computation failed: {e}", exc_info=True)
            return pd.DataFrame(index=dataset.index)

    def _modern_market_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Modern market structure features for post-2020 adaptation.
        
        These features capture:
        - Index concentration (passive flow effects)
        - Breadth divergence (narrow vs broad rallies)
        - Momentum crowding (reversal risk)
        - Passive flow patterns (month-end, SIP effects)
        - Volatility regime awareness
        
        Designed to help model adapt to changed market structure.
        """
        try:
            required = {"date", "close"}
            if not required.issubset(set(dataset.columns)):
                logger.warning(f"Modern market block: Missing columns. Have: {dataset.columns.tolist()[:5]}...")
                return pd.DataFrame(index=dataset.index)
            
            # Build modern features from price data
            modern_df = build_modern_features(
                dataset,
                date_col="date",
                ticker_col="ticker" if "ticker" in dataset.columns else None,
                close_col="close",
                volume_col="volume" if "volume" in dataset.columns else None,
            )
            
            if modern_df.empty:
                return pd.DataFrame(index=dataset.index)
            
            # Merge back to dataset on date
            if "date" in modern_df.columns:
                modern_df["date"] = pd.to_datetime(modern_df["date"])
                dataset_dates = pd.to_datetime(dataset["date"])
                
                # Create mapping from date to feature values
                modern_df = modern_df.set_index("date")
                
                # Reindex to match dataset
                out = pd.DataFrame(index=dataset.index)
                for col in modern_df.columns:
                    out[col] = dataset_dates.map(modern_df[col].to_dict())
                
                # Fill NaNs with 0 (or could ffill)
                out = out.fillna(0.0)
                
                logger.info(f"Modern market block: Added {len(out.columns)} features")
                return out
            else:
                return modern_df.reindex(dataset.index).fillna(0.0)
                
        except Exception as e:
            logger.warning(f"Modern market feature computation failed: {e}", exc_info=True)
            return pd.DataFrame(index=dataset.index)

    def _fii_dii_flow_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """FII/DII institutional flow features.
        
        Captures institutional buying/selling patterns from NSE data.
        FII/DII flows are strong predictors of short-term stock movements in India.
        
        Features:
        - fii_net_5d: 5-day rolling FII net buying (Rs Cr)
        - dii_net_5d: 5-day rolling DII net buying (Rs Cr)
        - fii_dii_ratio: FII net / DII net (institutional sentiment)
        - fii_momentum_10d: Change in FII net buying trend
        - fii_vs_market: FII flow direction vs market return correlation
        """
        if "date" not in dataset.columns:
            return pd.DataFrame(index=dataset.index)
        
        try:
            from pathlib import Path
            
            # Try to load cached FII/DII data
            cache_path = Path(settings.data_dir) / "raw" / "fii_dii" / "fii_dii_cache.parquet"
            
            if cache_path.exists():
                fii_data = pd.read_parquet(cache_path)
                fii_data['date'] = pd.to_datetime(fii_data['date'])
                # Convert numeric columns that may be stored as strings
                for num_col in ['buyValue', 'sellValue', 'netValue', 'fii_cash_net', 'dii_cash_net']:
                    if num_col in fii_data.columns:
                        fii_data[num_col] = pd.to_numeric(fii_data[num_col], errors='coerce')
                # Normalize category-pivot format → wide format with fii_cash_net/dii_cash_net
                if 'category' in fii_data.columns and 'fii_cash_net' not in fii_data.columns:
                    fii_pivot = fii_data.copy()
                    fii_pivot['category'] = fii_data['category'].str.strip().str.upper()
                    fii_mask = fii_pivot['category'].str.startswith('FII')
                    dii_mask = fii_pivot['category'].str.startswith('DII')
                    net_col = 'netValue' if 'netValue' in fii_pivot.columns else 'net_value'
                    fii_net = fii_pivot.loc[fii_mask, ['date', net_col]].rename(columns={net_col: 'fii_cash_net'})
                    dii_net = fii_pivot.loc[dii_mask, ['date', net_col]].rename(columns={net_col: 'dii_cash_net'})
                    fii_data = fii_net.merge(dii_net, on='date', how='outer').sort_values('date')
                if fii_data.empty or 'fii_cash_net' not in fii_data.columns or len(fii_data) < 5:
                    return pd.DataFrame(index=dataset.index)
            else:
                # Try to fetch from NSE
                try:
                    from intentflow_ai.data.providers.fii_dii_provider import get_fii_dii_provider
                    
                    provider = get_fii_dii_provider()
                    start = dataset['date'].min()
                    end = dataset['date'].max()
                    fii_data = provider.fetch_fii_dii_data(start, end)
                    
                    if fii_data.empty:
                        logger.warning("FII/DII data not available, returning empty features")
                        return pd.DataFrame(index=dataset.index)
                except Exception as e:
                    logger.warning(f"FII/DII provider not available: {e}")
                    return pd.DataFrame(index=dataset.index)
            
            # Compute rolling features on FII/DII data
            fii_data = fii_data.sort_values('date')
            
            # Rolling net buying
            fii_data['fii_net_5d'] = fii_data['fii_cash_net'].rolling(5).sum()
            fii_data['dii_net_5d'] = fii_data['dii_cash_net'].rolling(5).sum()
            fii_data['fii_net_10d'] = fii_data['fii_cash_net'].rolling(10).sum()
            fii_data['dii_net_10d'] = fii_data['dii_cash_net'].rolling(10).sum()
            
            # Institutional sentiment ratio
            fii_data['fii_dii_ratio'] = fii_data['fii_cash_net'] / (fii_data['dii_cash_net'].replace(0, 1e-9))
            
            # FII momentum (change in trend)
            fii_data['fii_momentum_10d'] = fii_data['fii_net_5d'].pct_change(5)
            
            # Z-scores of flows
            fii_data['fii_net_z'] = (fii_data['fii_cash_net'] - fii_data['fii_cash_net'].rolling(20).mean()) / (fii_data['fii_cash_net'].rolling(20).std() + 1e-9)
            fii_data['dii_net_z'] = (fii_data['dii_cash_net'] - fii_data['dii_cash_net'].rolling(20).mean()) / (fii_data['dii_cash_net'].rolling(20).std() + 1e-9)
            
            # Binary signals
            fii_data['fii_buying'] = (fii_data['fii_cash_net'] > 0).astype(float)
            fii_data['dii_buying'] = (fii_data['dii_cash_net'] > 0).astype(float)
            fii_data['both_buying'] = ((fii_data['fii_cash_net'] > 0) & (fii_data['dii_cash_net'] > 0)).astype(float)
            
            # Create output DataFrame
            out = pd.DataFrame(index=dataset.index)
            
            # Map features to dataset dates
            feature_cols = [
                'fii_net_5d', 'dii_net_5d', 'fii_net_10d', 'dii_net_10d',
                'fii_dii_ratio', 'fii_momentum_10d', 'fii_net_z', 'dii_net_z',
                'fii_buying', 'dii_buying', 'both_buying'
            ]
            
            fii_lookup = fii_data.set_index('date')
            dataset_dates = pd.to_datetime(dataset['date'])
            
            for col in feature_cols:
                if col in fii_lookup.columns:
                    out[col] = dataset_dates.map(fii_lookup[col].to_dict())
            
            # Fill NaN with 0
            out = out.fillna(0.0)
            
            logger.info(f"FII/DII flow block: Added {len(out.columns)} features")
            return out
            
        except Exception as e:
            logger.warning(f"FII/DII flow feature computation failed: {e}")
            return pd.DataFrame(index=dataset.index)

    def _global_overnight_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Global market overnight return features.

        US and Asian markets close before India opens. Their previous-day returns
        are among the most reliable short-term predictors of Indian market direction.

        Features:
        - sp500_ret_1d: S&P 500 previous day return
        - nasdaq_ret_1d: Nasdaq previous day return
        - asia_ret_1d: Average Nikkei + Hang Seng previous day return
        - gold_ret_5d: Gold 5-day return (safe haven indicator)
        - copper_ret_5d: Copper 5-day return (growth proxy)
        - dxy_ret_5d: USD strength proxy (USD/INR 5-day return)
        - us_risk_on: S&P up AND Nasdaq up (risk-on signal)
        - global_momentum_5d: Average of S&P + Nasdaq 5-day returns
        """
        if "date" not in dataset.columns:
            return pd.DataFrame(index=dataset.index)

        try:
            from intentflow_ai.data.macro_provider import MacroDataProvider

            provider = MacroDataProvider()

            dates = pd.to_datetime(dataset["date"].unique())
            if len(dates) == 0:
                return pd.DataFrame(index=dataset.index)

            # Pre-load all global tickers to cache
            global_tickers = [
                provider.config.sp500_ticker,
                provider.config.nasdaq_ticker,
                provider.config.nikkei_ticker,
                provider.config.hangseng_ticker,
                provider.config.gold_ticker,
                provider.config.copper_ticker,
            ]
            for t in global_tickers:
                provider._load_ticker_data(t)

            # Compute features for each unique date
            date_features = {}
            for dt in sorted(dates):
                feats = provider.get_global_features(dt)
                date_features[dt] = feats

            if not date_features:
                return pd.DataFrame(index=dataset.index)

            # Build lookup DataFrame
            feat_df = pd.DataFrame.from_dict(date_features, orient="index")
            feat_df.index = pd.to_datetime(feat_df.index)

            # Add derived features
            if "sp500_ret_1d" in feat_df.columns and "nasdaq_ret_1d" in feat_df.columns:
                feat_df["us_risk_on"] = (
                    (feat_df["sp500_ret_1d"] > 0) & (feat_df["nasdaq_ret_1d"] > 0)
                ).astype(float)
                feat_df["global_momentum_5d"] = (
                    feat_df["sp500_ret_1d"].rolling(5, min_periods=1).mean()
                    + feat_df["nasdaq_ret_1d"].rolling(5, min_periods=1).mean()
                ) / 2

            # Map to dataset rows
            out = pd.DataFrame(index=dataset.index)
            dataset_dates = pd.to_datetime(dataset["date"])

            for col in feat_df.columns:
                lookup = feat_df[col].to_dict()
                out[col] = dataset_dates.map(lookup)

            out = out.fillna(0.0)
            logger.info(f"Global overnight block: Added {len(out.columns)} features")
            return out

        except Exception as e:
            logger.warning(f"Global overnight feature computation failed: {e}")
            return pd.DataFrame(index=dataset.index)

    def _delivery_momentum_block(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Delivery × momentum interaction features.

        These capture the interplay between institutional delivery patterns and
        price momentum — delivery spikes concurrent with price moves suggest
        informed institutional activity.

        Features:
        - delivery_spike_x_momentum: delivery_z × sign(5d return) — confirms direction
        - accumulation_score: delivery_ratio_change × abs(10d return) — size of conviction
        - smart_money_divergence: delivery rising + price flat/falling — accumulation before breakout
        - distribution_signal: delivery falling + price rising — distribution before drop
        - delivery_breakout_confirm: delivery spike + strong positive momentum
        """
        required = {"close", "date"}
        if not required.issubset(set(dataset.columns)):
            return pd.DataFrame(index=dataset.index)

        out = pd.DataFrame(index=dataset.index)

        try:
            close = dataset["close"].astype(float)
            ret_5d = close.pct_change(5)
            ret_10d = close.pct_change(10)

            # Check if delivery data is available
            has_delivery = "delivery_qty" in dataset.columns or "delivery_ratio" in dataset.columns

            if has_delivery:
                # Compute delivery z-score
                if "delivery_qty" in dataset.columns:
                    dq = dataset["delivery_qty"].astype(float)
                    dq_mean = dq.rolling(20, min_periods=5).mean()
                    dq_std = dq.rolling(20, min_periods=5).std().replace(0, 1e-9)
                    delivery_z = (dq - dq_mean) / dq_std
                else:
                    delivery_z = pd.Series(0.0, index=dataset.index)

                if "delivery_ratio" in dataset.columns:
                    dr = dataset["delivery_ratio"].astype(float)
                    dr_change_10 = dr.pct_change(10)
                else:
                    dr_change_10 = pd.Series(0.0, index=dataset.index)

                # delivery_z × sign(5d return) — confirms direction
                out["delivery_spike_x_momentum"] = delivery_z * np.sign(ret_5d)

                # delivery_ratio_change × abs(10d return)
                out["accumulation_score"] = dr_change_10 * ret_10d.abs()

                # Smart money divergence: delivery rising but price flat/falling
                delivery_rising = (delivery_z > 1.0).astype(float)
                price_flat_down = (ret_5d <= 0.01).astype(float)  # flat or down
                out["smart_money_divergence"] = delivery_rising * price_flat_down

                # Distribution: delivery falling but price rising
                delivery_falling = (delivery_z < -1.0).astype(float)
                price_rising = (ret_5d > 0.01).astype(float)
                out["distribution_signal"] = delivery_falling * price_rising

                # Delivery breakout confirm: strong delivery + strong momentum
                out["delivery_breakout_confirm"] = (
                    (delivery_z > 1.5).astype(float) * (ret_5d > 0.03).astype(float)
                )
            else:
                # No delivery data — still provide pure momentum interactions
                out["momentum_acceleration"] = ret_5d - ret_10d / 2
                vol = dataset["volume"].astype(float) if "volume" in dataset.columns else pd.Series(1.0, index=dataset.index)
                vol_z = (vol - vol.rolling(20, min_periods=5).mean()) / (vol.rolling(20, min_periods=5).std() + 1e-9)
                out["volume_momentum_confirm"] = vol_z * np.sign(ret_5d)

            out = out.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            logger.info(f"Delivery-momentum block: Added {len(out.columns)} features")
            return out

        except Exception as e:
            logger.warning(f"Delivery-momentum feature computation failed: {e}")
            return pd.DataFrame(index=dataset.index)

