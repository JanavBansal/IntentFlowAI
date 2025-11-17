"""Simple top-K holding-period backtest utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from intentflow_ai.backtest.filters import MetaFilterConfig, RiskFilterConfig, apply_cooldown, compute_regime_flags, update_cooldown


@dataclass
class BacktestConfig:
    date_col: str = "date"
    ticker_col: str = "ticker"
    close_col: str = "close"
    proba_col: str = "proba"
    label_col: str = "label"
    hold_days: int = 10
    top_k: int = 10
    max_weight: float = 0.10
    slippage_bps: float = 10.0
    fee_bps: float = 1.0
    rebalance: str = "daily"
    long_only: bool = True
    risk: RiskFilterConfig = field(default_factory=RiskFilterConfig)
    meta: MetaFilterConfig = field(default_factory=MetaFilterConfig)


def backtest_signals(preds: pd.DataFrame, prices: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, object]:
    """Run a basic ranked-probability backtest."""

    if cfg.rebalance != "daily":
        raise ValueError("Only daily rebalance supported for now.")

    preds = preds.copy()
    prices = prices.copy()
    preds[cfg.date_col] = pd.to_datetime(preds[cfg.date_col])
    prices[cfg.date_col] = pd.to_datetime(prices[cfg.date_col])
    preds = preds.dropna(subset=[cfg.proba_col])

    px = prices.pivot_table(index=cfg.date_col, columns=cfg.ticker_col, values=cfg.close_col)
    if px.empty:
        return _empty_backtest(cfg)

    regimes = compute_regime_flags(px, cfg.risk)
    dates = sorted(preds[cfg.date_col].unique())
    k = max(1, int(cfg.top_k))
    cost_mult_in = 1.0 + (cfg.slippage_bps + cfg.fee_bps) / 1e4
    cost_mult_out = 1.0 - (cfg.slippage_bps + cfg.fee_bps) / 1e4
    cooldown_state: Dict[str, pd.Timestamp] = {}
    active_positions: list[Dict[str, object]] = []
    trades: list[dict] = []
    prev_ranks: Optional[Dict[str, int]] = None
    for d in dates:
        d_ts = pd.to_datetime(d)
        active_positions = [pos for pos in active_positions if pos["date_out"] > d_ts]
        if d_ts not in regimes.index:
            continue
        if not bool(regimes.loc[d_ts, "allow_entry"]):
            continue
        if d not in px.index:
            continue
        day_preds = preds.loc[preds[cfg.date_col] == d].sort_values(cfg.proba_col, ascending=False)
        if cfg.meta.enabled and cfg.meta.proba_col in day_preds.columns:
            day_preds = day_preds[day_preds[cfg.meta.proba_col] >= cfg.meta.min_prob]
        ticker_list = apply_cooldown(day_preds[cfg.ticker_col].tolist(), cooldown_state, d_ts)
        if cfg.risk.max_positions:
            capacity = max(cfg.risk.max_positions - len(active_positions), 0)
            if capacity <= 0:
                continue
            k_today = min(k, capacity)
        else:
            k_today = k
        ranks_today = {ticker: rank + 1 for rank, ticker in enumerate(ticker_list)}
        picks = _stable_topk(ranks_today, prev_ranks, k_today, max_drop=20)
        prev_ranks = ranks_today
        cols = [t for t in picks if t in px.columns]
        if not cols:
            continue
        entry_px = px.loc[d, cols].dropna()
        if entry_px.empty:
            continue
        exit_idx = px.index.get_indexer([d])[0] + cfg.hold_days
        if exit_idx >= len(px.index):
            continue
        d_out = px.index[exit_idx]
        exit_px = px.loc[d_out, entry_px.index].dropna()
        if exit_px.empty:
            continue

        valid = entry_px.index.intersection(exit_px.index)
        if valid.empty:
            continue

        entry = entry_px[valid] * cost_mult_in
        exit_ = exit_px[valid] * cost_mult_out
        gross = (exit_ / entry) - 1.0

        weights = np.full(len(valid), min(1.0 / len(valid), cfg.max_weight))

        for tkr, gr in gross.items():
            update_cooldown(cooldown_state, [tkr], d_ts, cfg.risk.cooldown_days)
            active_positions.append({"ticker": tkr, "date_out": d_out})
            trades.append(
                {
                    "date_in": d,
                    "date_out": d_out,
                    "ticker": tkr,
                    "entry_px": float(entry[tkr]),
                    "exit_px": float(exit_[tkr]),
                    "gross_ret": float(gr),
                    "net_ret": float(gr),
                }
            )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return _empty_backtest(cfg)

    daily = trades_df.groupby("date_in")["net_ret"].mean().reindex(px.index, fill_value=0.0)
    equity = (1.0 + daily).cumprod()
    ret_daily = daily.values
    ann = 252
    length = max(len(daily), 1)
    if equity.empty:
        cagr = 0.0
    else:
        cagr = float(equity.iloc[-1] ** (ann / length) - 1.0)
    std = float(np.std(ret_daily))
    sharpe = float(np.mean(ret_daily) / (std + 1e-12) * np.sqrt(ann)) if len(ret_daily) > 1 else 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    maxdd = float(dd.min()) if not dd.empty else 0.0
    win_rate = float((trades_df["net_ret"] > 0).mean())
    turnover = float(len(trades_df) / length)
    avg_positions = float(trades_df.groupby("date_in").size().mean()) if not trades_df.empty else 0.0

    summary = {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "maxDD": maxdd,
        "turnover": turnover,
        "win_rate": win_rate,
        "avg_hold_days": float(cfg.hold_days),
        "avg_positions": avg_positions,
    }

    equity.name = "equity"
    return {"equity_curve": equity, "trades": trades_df, "summary": summary}


def _empty_backtest(cfg: BacktestConfig) -> Dict[str, object]:
    return {
        "equity_curve": pd.Series(dtype=float, name="equity"),
        "trades": pd.DataFrame(
            columns=["date_in", "date_out", "ticker", "entry_px", "exit_px", "gross_ret", "net_ret"]
        ),
        "summary": {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "maxDD": 0.0,
            "turnover": 0.0,
            "win_rate": 0.0,
            "avg_hold_days": float(cfg.hold_days),
            "avg_positions": 0.0,
        },
    }


def _stable_topk(ranks_today: Dict[str, int], ranks_prev: Optional[Dict[str, int]], k: int, max_drop: int = 20):
    """Keep prior winners unless they fall sharply, then fill with today's best."""

    ranks_prev = ranks_prev or {}
    keep = [
        t
        for t, prev_rank in ranks_prev.items()
        if prev_rank <= k and t in ranks_today and ranks_today[t] <= prev_rank + max_drop
    ]
    keep_set = set(keep)
    ordered_today = sorted(ranks_today.items(), key=lambda x: x[1])
    add = [t for t, _ in ordered_today if t not in keep_set][: max(0, k - len(keep))]
    return keep + add
