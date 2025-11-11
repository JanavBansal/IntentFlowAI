"""Ingestion orchestration for IntentFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.data.sources import DEFAULT_SOURCE_REGISTRY, SourceRegistry
from intentflow_ai.utils.io import write_parquet_partition
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataIngestionWorkflow:
    """Coordinates fetching data from multiple logical sources.

    The workflow pulls from each registered source, applies lightweight
    validation or transformation hooks, and writes outputs into the local
    `data/` lake where downstream feature engineering jobs can read them.
    """

    output_dir: Path = field(default_factory=lambda: settings.data_dir)
    registry: SourceRegistry = field(default_factory=lambda: DEFAULT_SOURCE_REGISTRY)
    cfg: Settings = field(default_factory=lambda: settings)
    validators: Mapping[str, callable] = field(default_factory=dict)

    def run(self) -> None:
        logger.info("Starting ingestion run", extra={"output_dir": str(self.output_dir)})
        for source_name in self.registry.factories:
            self._ingest_source(source_name)

    def _ingest_source(self, source_name: str) -> None:
        source = self.registry.build(source_name)
        logger.info("Fetching records", extra={"source": source.name})
        records = source.fetch()
        self._write_records(source.name, records)

    def _write_records(self, name: str, records: Iterable[dict]) -> None:
        target = self.output_dir / "raw" / name
        logger.debug("Persisting batch", extra={"target": str(target)})
        write_parquet_partition(target, records)


__all__ = ["DataIngestionWorkflow"]
