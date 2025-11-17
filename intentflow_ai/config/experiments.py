"""Experiment configuration loader and helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from intentflow_ai.config.settings import Settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentConfig:
    """Thin wrapper for parsed experiment YAML."""

    def __init__(self, path: Path, data: Dict[str, Any]):
        self.path = path
        self.data = data
        self.id = data.get("id") or path.stem

    def get(self, key: str, default=None):
        return self.data.get(key, default)


def load_experiment_config(path: str | Path | None) -> Optional[ExperimentConfig]:
    """Load experiment YAML into a structured object."""

    if path is None:
        return None
    exp_path = Path(path)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    data = yaml.safe_load(exp_path.read_text()) or {}
    cfg = ExperimentConfig(exp_path, data)
    logger.info("Loaded experiment config", extra={"id": cfg.id, "path": str(exp_path)})
    return cfg


def apply_experiment_overrides(base: Settings, exp: ExperimentConfig | None) -> Settings:
    """Return a Settings copy overridden by the YAML experiment values."""

    if exp is None:
        return base

    overrides: Dict[str, Any] = {}
    splits = exp.get("splits", {})
    runtime = exp.get("runtime", {})
    universe_block = exp.get("universe", {})

    if "train_start" in splits:
        overrides["price_start"] = splits["train_start"]
    if "train_end" in splits:
        overrides["price_end"] = splits["train_end"]
    if "valid_start" in splits:
        overrides["valid_start"] = splits["valid_start"]
    if "test_start" in splits:
        overrides["test_start"] = splits["test_start"]

    if "random_state" in exp.data:
        overrides["lgbm_seed"] = exp.data["random_state"]
        # Also seed LightGBM defaults for determinism
        new_lgbm = replace(base.lgbm, random_state=exp.data["random_state"])
        overrides["lgbm"] = new_lgbm

    if "min_tickers" in runtime:
        overrides["min_train_tickers"] = runtime["min_tickers"]

    if universe_block:
        if "universe_file" in universe_block:
            overrides["universe_file"] = universe_block["universe_file"]
        if universe_block.get("use_membership") is False or universe_block.get("membership_file") is None:
            overrides["universe_membership_file"] = ""
        elif "membership_file" in universe_block and universe_block["membership_file"]:
            overrides["universe_membership_file"] = universe_block["membership_file"]

    backtest_block = exp.get("backtest")
    if backtest_block:
        backtest_defaults = replace(
            base.backtest,
            top_k=backtest_block.get("top_k", base.backtest.top_k),
            hold_days=backtest_block.get("hold_days", base.backtest.hold_days),
            slippage_bps=backtest_block.get("slippage_bps", base.backtest.slippage_bps),
            fee_bps=backtest_block.get("fee_bps", base.backtest.fee_bps),
        )
        overrides["backtest"] = backtest_defaults

    if not overrides:
        return base

    return replace(base, **overrides)


def backtest_params_from_experiment(exp: ExperimentConfig | None) -> Dict[str, Any]:
    """Extract backtest-specific knobs from the experiment YAML."""

    if exp is None:
        return {}
    backtest_block = exp.get("backtest", {})
    risk_filters = exp.get("risk_filters", {})
    meta_config = exp.get("meta_labeling", {})
    return {
        "top_k": backtest_block.get("top_k"),
        "hold_days": backtest_block.get("hold_days"),
        "slippage_bps": backtest_block.get("slippage_bps"),
        "fee_bps": backtest_block.get("fee_bps"),
        "cost_model": backtest_block.get("cost_model"),
        "risk_filters": risk_filters,
        "meta_labeling": meta_config,
    }
