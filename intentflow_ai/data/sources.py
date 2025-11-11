"""Source adaptors for ownership, transactions, fundamentals, narratives, and prices."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Protocol, Sequence

import pandas as pd

from intentflow_ai.config.settings import settings
from intentflow_ai.data.universe import load_universe
from intentflow_ai.utils.cache import cache_parquet
from intentflow_ai.utils.contracts import validate_schema
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)

def _load_universe_df() -> pd.DataFrame:
    path = settings.data_dir / Path(settings.universe_file)
    try:
        return load_universe(path)
    except Exception as exc:  # pragma: no cover - config issues surfaced at runtime
        logger.error("Failed to load universe file", extra={"error": str(exc), "path": str(path)})
        return pd.DataFrame(columns=["ticker_nse", "ticker_yf", "sector"])


UNIVERSE_DF = _load_universe_df()

PRICE_SCHEMA = {
    "date": "datetime64[ns]",
    "open": "float",
    "high": "float",
    "low": "float",
    "close": "float",
    "volume": "float",
    "ticker": "string",
    "sector": "string",
}


class DataSource(Protocol):
    """Lightweight protocol each ingestion adaptor should satisfy."""

    name: str

    def fetch(self, *args, **kwargs) -> Iterable[dict]:
        """Return an iterable of normalized records."""


@dataclass
class SourceRegistry:
    """Registry mapping logical names to callables that build data sources.

    This provides a centralized lookup for ingestion workflows and makes it
    easy to swap implementations (e.g., switch between mock CSV readers and
    production APIs) without touching orchestration code.
    """

    factories: Dict[str, Callable[[], DataSource]]

    def build(self, name: str) -> DataSource:
        if name not in self.factories:
            raise KeyError(f"Unknown data source: {name}")
        return self.factories[name]()


def placeholder_source(name: str) -> DataSource:
    """Return a stub source that documents future integration points."""

    class _Stub:
        def __init__(self, source_name: str) -> None:
            self.name = source_name

        def fetch(self, *args, **kwargs):  # type: ignore[override]
            raise NotImplementedError(
                f"{self.name} source is not wired yet. Implement fetch() to connect "
                "to live data (REST, database, vendor file drops, etc.)."
            )

    return _Stub(name)


class PriceCSVSource:
    """Adapter that reads OHLCV CSVs from the local raw lake."""

    REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume", "sector"}

    def __init__(self, root_dir: Path, subdir: str = "prices") -> None:
        self.root_dir = Path(root_dir)
        self.subdir = subdir
        self.name = "price_confirmation"

    def fetch(self) -> Iterable[dict]:
        source_dir = self.root_dir / "raw" / self.subdir
        rows: list[dict] = []
        if source_dir.exists():
            for csv_path in sorted(source_dir.glob("*.csv")):
                rows.extend(self._read_file(csv_path))
        if not rows:
            rows.append(
                {
                    "date": None,
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "close": 0.0,
                    "volume": 0.0,
                    "sector": "unknown",
                    "ticker": "PLACEHOLDER",
                }
            )
        return rows

    def _read_file(self, path: Path) -> list[dict]:
        frame = pd.read_csv(path)
        frame.columns = [col.strip().lower() for col in frame.columns]
        missing = self.REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame["ticker"] = path.stem.upper()
        cols = ["date", "open", "high", "low", "close", "volume", "sector", "ticker"]
        return frame[cols].to_dict("records")


@dataclass
class PriceAPISource:
    """Download OHLCV from Yahoo Finance for NSE tickers."""

    universe: pd.DataFrame
    data_dir: Path
    name: str = "price_confirmation"

    def __post_init__(self) -> None:
        self.start_date = settings.price_start
        self.end_date = settings.price_end
        self.min_trading_days = settings.min_trading_days
        self.cache_path = self.data_dir / "raw" / "price_confirmation" / "yf_cache.parquet"

    def fetch(self) -> Iterable[dict]:
        if self.universe.empty:
            logger.warning("Universe is empty; no prices fetched.")
            return []

        key = f"{self.start_date}_{self.end_date}_{len(self.universe)}"
        df = cache_parquet(self.cache_path, key, self._download_prices)
        df = validate_schema(df, PRICE_SCHEMA)
        df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])

        counts = df.groupby("ticker")["date"].transform("count")
        mask = counts >= self.min_trading_days
        dropped = df.loc[~mask, "ticker"].unique()
        if len(dropped):
            logger.warning(
                "Dropping tickers below min_trading_days",
                extra={"tickers": dropped.tolist(), "threshold": self.min_trading_days},
            )
        df = df.loc[mask]
        return df.to_dict("records")

    def _download_prices(self) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("yfinance is required for PriceAPISource. Install yfinance>=0.2.0") from exc

        frames = []
        for _, row in self.universe.iterrows():
            yf_symbol = row["ticker_yf"]
            ticker = row["ticker_nse"]
            sector = row["sector"]
            data = self._fetch_symbol(yf, yf_symbol)
            if data is None:
                continue
            data = data.reset_index().rename(columns=str.lower)
            subset = data[["date", "open", "high", "low", "close", "volume"]].copy()
            subset["ticker"] = ticker
            subset["sector"] = sector
            frames.append(subset)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame(columns=list(PRICE_SCHEMA.keys()))

    def _fetch_symbol(self, yf_module, symbol: str) -> pd.DataFrame | None:
        end = self.end_date or None
        for attempt in range(3):
            try:
                data = yf_module.download(
                    symbol,
                    start=self.start_date,
                    end=end,
                    progress=False,
                    group_by="column",
                    auto_adjust=False,
                )
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(-1)
                if data.empty:
                    return None
                return data
            except Exception as exc:  # pragma: no cover - network noise
                delay = 2 ** attempt
                logger.warning(
                    "Price download failed; retrying",
                    extra={"symbol": symbol, "attempt": attempt + 1, "delay": delay, "error": str(exc)},
                )
                time.sleep(delay)
        logger.error("Giving up on symbol", extra={"symbol": symbol})
        return None


@dataclass
class OwnershipSource:
    """Fetch FII/DII participation data from NSE API or cached CSV."""

    source_file: Path | None = None
    name: str = "ownership_flows"

    def fetch(self) -> Iterable[dict]:
        if self.source_file and self.source_file.exists():
            df = pd.read_csv(self.source_file)
        else:
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=5, freq="B"),
                    "ticker": ["MOCK"] * 5,
                    "fii_hold": [0.12, 0.125, 0.13, 0.131, 0.135],
                    "dii_hold": [0.22, 0.221, 0.223, 0.225, 0.227],
                }
            )
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df.to_dict("records")


@dataclass
class DeliverySource:
    """Parse security-wise delivery CSVs from NSE."""

    directory: Path | None = None
    name: str = "delivery_transactions"

    def fetch(self) -> Iterable[dict]:
        rows: List[dict] = []
        if self.directory and self.directory.exists():
            for csv_path in sorted(self.directory.glob("*.csv")):
                df = pd.read_csv(csv_path)
                df.columns = [c.strip().lower() for c in df.columns]
                if not {"tradingsymbol", "deliveryqty", "tradedqty", "date"}.issubset(df.columns):
                    continue
                df = df.rename(columns={"tradingsymbol": "ticker"})
                df["delivery_ratio"] = df["deliveryqty"] / df["tradedqty"].replace(0, pd.NA)
                rows.extend(df.to_dict("records"))
        if not rows:
            rows = [
                {
                    "date": "2024-01-01",
                    "ticker": "MOCK",
                    "delivery_ratio": 0.42,
                    "deliveryqty": 100000,
                    "tradedqty": 240000,
                }
            ]
        return rows


@dataclass
class FundamentalSource:
    """Load EPS/revenue metrics from Screener export or CSV."""

    csv_path: Path | None = None
    name: str = "fundamental_drift"

    def fetch(self) -> Iterable[dict]:
        if self.csv_path and self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
        else:
            df = pd.DataFrame(
                {
                    "ticker": ["MOCK"],
                    "report_date": ["2024-03-31"],
                    "eps": [10.5],
                    "revenue": [1_000_000_000],
                    "pe": [22.5],
                }
            )
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")
        return df.to_dict("records")


@dataclass
class NarrativeSource:
    """Fetch news headlines and sentiment scores."""

    api_key: str | None = None
    symbols: List[str] = field(default_factory=list)
    name: str = "narrative_tone"

    def fetch(self) -> Iterable[dict]:
        rows: List[dict] = []
        if not self.symbols:
            self.symbols = ["RELIANCE"]
        for symbol in self.symbols:
            rows.append(
                {
                    "ticker": symbol,
                    "headline": f"{symbol} sentiment placeholder",
                    "sentiment": 0.1,
                    "published_at": pd.Timestamp.utcnow().isoformat(),
                    "source": "placeholder",
                }
            )
        return rows


DEFAULT_SOURCE_REGISTRY = SourceRegistry(
    factories={
        "ownership": lambda: OwnershipSource(),
        "transactions": lambda: DeliverySource(),
        "fundamentals": lambda: FundamentalSource(),
        "narrative": lambda: NarrativeSource(),
        "price": lambda: PriceAPISource(universe=UNIVERSE_DF, data_dir=settings.data_dir),
    }
)
"""Starter registry covering the five signal layers described in the spec."""
