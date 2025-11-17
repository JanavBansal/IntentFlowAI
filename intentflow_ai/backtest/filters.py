"""Reusable risk and regime filters for the backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass
class RiskFilterConfig:
    """Price-based regime controls."""

    trend_fast: int = 50
    trend_slow: int = 200
    vol_lookback: int = 20
    vol_high: float = 0.04
    allow_high_vol: bool = False
    allow_downtrend: bool = False
    max_positions: int | None = None
    cooldown_days: int = 0


@dataclass
class MetaFilterConfig:
    """Meta-model gating config."""

    enabled: bool = False
    proba_col: str = "meta_proba"
    min_prob: float = 0.5


def compute_regime_flags(px: pd.DataFrame, cfg: RiskFilterConfig) -> pd.DataFrame:
    """Return per-date flags for trend/volatility regimes."""

    idx = px.mean(axis=1)
    ma_fast = idx.rolling(cfg.trend_fast).mean()
    ma_slow = idx.rolling(cfg.trend_slow).mean()
    trend_ok = ma_fast > ma_slow

    ret = idx.pct_change()
    vol = ret.rolling(cfg.vol_lookback).std().fillna(0)
    vol_ok = vol < cfg.vol_high if cfg.vol_high > 0 else pd.Series(True, index=idx.index)

    allow = trend_ok | cfg.allow_downtrend
    if not cfg.allow_high_vol:
        allow = allow & vol_ok

    return pd.DataFrame(
        {
            "trend_ok": trend_ok,
            "vol_ok": vol_ok,
            "allow_entry": allow,
            "index_vol": vol,
        }
    )


def apply_cooldown(tickers: Iterable[str], cooldown_state: Dict[str, pd.Timestamp], current: pd.Timestamp) -> List[str]:
    """Filter tickers that are still in cooldown."""

    allowed = []
    for t in tickers:
        until = cooldown_state.get(t)
        if until is None or until <= current:
            allowed.append(t)
    return allowed


def update_cooldown(cooldown_state: Dict[str, pd.Timestamp], tickers: Iterable[str], current: pd.Timestamp, days: int) -> None:
    """Mark tickers as on cooldown until `current + days`."""

    if days <= 0:
        return
    for t in tickers:
        cooldown_state[t] = current + pd.Timedelta(days=days)
