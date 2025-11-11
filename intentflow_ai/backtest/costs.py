"""Cost model helpers for the backtest."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def load_cost_model(name: str, path: Path | None = None) -> Dict[str, object]:
    """Load a named transaction-cost model from YAML and compute total bps."""

    cost_path = Path(path) if path else settings.path("config/costs_india.yaml")
    if not cost_path.exists():
        raise FileNotFoundError(f"Cost config not found at {cost_path}")
    data = yaml.safe_load(cost_path.read_text()) or {}
    models = data.get("models", {})
    default_model = data.get("default_model")
    model = models.get(name) or models.get(default_model)
    if model is None:
        raise KeyError(f"Unknown cost model '{name}'. Available: {list(models.keys())}")

    components = model.get("components", {})
    normalized = {}
    total = float(model.get("per_side_bps", 0.0))
    for key, value in components.items():
        try:
            normalized[key] = float(value)
            total += float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Component '{key}' in cost model '{name}' is not numeric.") from exc
    if total <= 0:
        raise ValueError(f"Cost model '{name}' resulted in non-positive total bps.")

    slippage_bps = float(model.get("slippage_bps", data.get("defaults", {}).get("slippage_bps", 0.0)))

    logger.info("Loaded cost model", extra={"cost_model": name, "total_bps": total, "slippage_bps": slippage_bps})
    return {
        "name": name,
        "components": normalized,
        "total_bps": float(total),
        "slippage_bps": slippage_bps,
        "description": model.get("description", ""),
        "source": str(cost_path),
    }
