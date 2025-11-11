"""Source adaptors for ownership, transactions, fundamentals, narratives, and prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Protocol


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


DEFAULT_SOURCE_REGISTRY = SourceRegistry(
    factories={
        "ownership": lambda: placeholder_source("ownership_flows"),
        "transactions": lambda: placeholder_source("delivery_transactions"),
        "fundamentals": lambda: placeholder_source("fundamental_drift"),
        "narrative": lambda: placeholder_source("narrative_tone"),
        "price": lambda: placeholder_source("price_confirmation"),
    }
)
"""Starter registry covering the five signal layers described in the spec."""
