"""Centralized configuration management.

The Settings dataclass holds frequently referenced configuration knobs such
as data directories, horizon parameters, LightGBM defaults, and feature flags.
In production we can extend this with environment variable reading,
config files (YAML/TOML), or secret managers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(slots=True)
class LightGBMConfig:
    """Defaults passed to the LightGBM trainer."""

    objective: str = "binary"
    metric: str = "auc"
    boosting_type: str = "gbdt"
    n_estimators: int = 1000
    learning_rate: float = 0.01
    num_leaves: int = 32  # Reduced from 64 to prevent overfitting
    max_depth: int = 5    # Constrained depth
    min_child_samples: int = 100  # Increased to require more data per leaf
    subsample: float = 0.8
    colsample_bytree: float = 0.6  # Feature subsampling
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


@dataclass
class BacktestDefaults:
    top_k: int = 10
    hold_days: int = 10
    slippage_bps: float = 10.0
    fee_bps: float = 1.0


@dataclass(slots=True)
class Settings:
    """Global configuration container for the project."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    experiments_dir: Path = project_root / "experiments"
    raw_sources: Dict[str, str] = field(
        default_factory=lambda: {
            "nse_bhavcopy": "s3://intentflow/raw/nse/bhavcopy/",
            "fiidii": "s3://intentflow/raw/ownership/",
            "newswire": "s3://intentflow/raw/news/",
        }
    )
    trading_universe: str = "nifty_200"
    universe_file: str = "external/universe/nifty100_universe.csv"
    universe_membership_file: str = "external/universe/nifty200_history.csv"
    signal_horizon_days: int = 10
    target_excess_return: float = 0.015
    price_start: str = "2018-01-01"
    price_end: str | None = None
    min_trading_days: int = 250  # Lowered from 600 to get more ticker coverage
    min_train_tickers: int = 180  # Restored to production guard - requires universe with ≥180 tickers
    cv_splits: int = 3
    valid_start: str = "2023-07-01"
    test_start: str = "2024-01-01"
    lgbm_seed: int = 42
    backtest: BacktestDefaults = field(default_factory=BacktestDefaults)
    lgbm: LightGBMConfig = field(default_factory=LightGBMConfig)

    def path(self, *parts: str) -> Path:
        """Return a path relative to the project root."""

        return self.project_root.joinpath(*parts)


settings = Settings()
"""Singleton-style settings instance for convenience imports."""
