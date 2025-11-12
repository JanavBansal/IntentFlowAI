"""API-based price fetching utilities (yfinance wrapper)."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yfinance as yf

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)

os.environ.setdefault("YFINANCE_CACHE_DIR", str(Path(".yfcache").resolve()))
Path(os.environ["YFINANCE_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
YF_KW = dict(auto_adjust=False, actions=False, progress=False, threads=True)
_SYNTH_SUFFIX = re.compile(r"_S\d+$")


@dataclass
class PriceFetchConfig:
    universe_path: Path
    output_csv: Path
    start: str = "2017-01-01"
    end: str | None = None
    suffix: str = ".NS"


def _clean_ticker(t: str) -> str:
    t = str(t).strip().upper()
    t = _SYNTH_SUFFIX.sub("", t)
    t = t.replace(".NS", "")
    # keep alphanumerics plus hyphen/ampersand (Yahoo tickers like BAJAJ-AUTO, M&M)
    t = re.sub(r"[^A-Z0-9&-]", "", t)
    return t


def load_universe(path: Path) -> List[str]:
    df = pd.read_csv(path)
    for col in ["ticker", "symbol", "TICKER", "SYMBOL"]:
        if col in df.columns:
            series = df[col].astype(str).map(_clean_ticker)
            tickers = series[series.str.len() > 0].drop_duplicates().tolist()
            return tickers
    raise ValueError("Universe CSV must include a ticker/symbol column.")


def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _normalize_batch(df: pd.DataFrame, batch: List[str]) -> pd.DataFrame:
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    if isinstance(df.columns, pd.MultiIndex):
        out = []
        for ticker in batch:
            sym = f"{ticker}.NS"
            if sym not in df.columns.levels[1]:
                sym = ticker
                if sym not in df.columns.levels[1]:
                    continue
            sub = df.xs(sym, axis=1, level=1, drop_level=False).droplevel(1, axis=1)
            sub = sub.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
            sub = sub.reset_index().rename(columns={"index": "date", "Date": "date"})
            sub["ticker"] = ticker
            sub = sub[["date", "ticker", "open", "high", "low", "close", "volume"]]
            out.append(sub)
        if not out:
            return pd.DataFrame(columns=cols)
        result = pd.concat(out, ignore_index=True)
        result.columns = cols
        return result
    else:
        tmp = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
        tmp["ticker"] = batch[0]
        tmp = tmp[["date", "ticker", "open", "high", "low", "close", "volume"]]
        return tmp


def fetch_and_save(cfg: PriceFetchConfig) -> None:
    universe = sorted(set(load_universe(cfg.universe_path)))
    logger.info("Universe tickers", extra={"count": len(universe)})

    frames = []
    for batch in chunked(universe, 20):
        symbols = [f"{t}{cfg.suffix}" for t in batch]
        tries = 0
        while True:
            try:
                df = yf.download(symbols, start=cfg.start, end=cfg.end, **YF_KW)
                part = _normalize_batch(df, batch)
                if part.empty:
                    logger.warning("Empty batch", extra={"batch": batch[:3]})
                else:
                    frames.append(part)
                break
            except Exception as exc:
                tries += 1
                if tries >= 4:
                    logger.error("Batch failed", extra={"batch": batch[:3], "error": str(exc)})
                    break
                sleep = 1.5 * tries
                logger.warning("Retrying batch", extra={"batch": batch[:3], "attempt": tries, "sleep": sleep})
                time.sleep(sleep)

    if not frames:
        raise RuntimeError("No data fetched. Check universe and API connectivity.")

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    result = result.dropna(subset=["close"]).drop_duplicates(subset=["date", "ticker"])
    result = result.sort_values(["date", "ticker"])

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cfg.output_csv, index=False)
    logger.info(
        "Wrote API prices",
        extra={
            "path": str(cfg.output_csv),
            "rows": len(result),
            "tickers": result["ticker"].nunique(),
            "span": f"{result['date'].min().date()}→{result['date'].max().date()}",
        },
    )


__all__ = ["PriceFetchConfig", "fetch_and_save"]
