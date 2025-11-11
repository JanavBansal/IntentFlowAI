"""One-button smoke test that verifies ingestion and training loops."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings
from intentflow_ai.data.ingestion import DataIngestionWorkflow
from intentflow_ai.data.sources import PriceCSVSource, SourceRegistry


def write_sample_prices(root: Path) -> None:
    """Create deterministic CSVs for two tickers under data/raw/prices."""

    root.mkdir(parents=True, exist_ok=True)
    tickers = {"SMOKEA": "financials", "SMOKEB": "financials"}
    dates = pd.date_range("2024-01-01", periods=50, freq="B")

    for ticker, sector in tickers.items():
        base_price = 100 + (5 if ticker.endswith("B") else 0)
        drift = 0.05 if ticker.endswith("A") else 0.4
        trend = pd.Series(range(len(dates)), index=dates) * drift
        close = base_price + trend
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000 + trend * 1_000,
                "sector": sector,
            }
        )
        csv_path = root / f"{ticker}.csv"
        df.to_csv(csv_path, index=False)


def run_training_script(repo_root: Path) -> None:
    """Execute the training CLI so artifacts land in experiments/v0."""

    cmd = [sys.executable, "scripts/run_training.py"]
    subprocess.run(cmd, check=True, cwd=repo_root)


def verify_metrics(repo_root: Path) -> None:
    metrics_path = repo_root / "experiments" / "v0" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    required = {"roc_auc", "pr_auc"}
    missing = required - metrics.keys()
    if missing:
        raise KeyError(f"Metrics missing keys: {', '.join(sorted(missing))}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prices_dir = settings.data_dir / "raw" / "prices"

    write_sample_prices(prices_dir)

    price_registry = SourceRegistry(factories={"price": lambda: PriceCSVSource(root_dir=settings.data_dir)})
    workflow = DataIngestionWorkflow(registry=price_registry)
    workflow.run()

    run_training_script(repo_root)
    verify_metrics(repo_root)

    print("smoke test passed")


if __name__ == "__main__":
    main()
