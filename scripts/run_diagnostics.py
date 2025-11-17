"""Run diagnostics on backtest results: per-year and regime-based metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.backtest.filters import RiskFilterConfig
from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.config.settings import Settings, settings
from intentflow_ai.diagnostics import DiagnosticsConfig, DiagnosticsRunner
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostics on backtest results")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name under experiments/ (e.g., v_universe_sanity)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML config (optional, for loading price data)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots",
    )
    parser.add_argument(
        "--vol-buckets",
        type=float,
        nargs="+",
        default=[0, 33, 67, 100],
        help="Volatility percentile buckets for regime classification (default: 0 33 67 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load experiment config if provided
    exp_cfg = None
    if args.config:
        exp_cfg = load_experiment_config(args.config)
        cfg = apply_experiment_overrides(Settings(), exp_cfg)
        import intentflow_ai.config.settings as settings_module

        settings_module.settings = cfg
    else:
        cfg = settings

    experiment_name = args.experiment
    exp_dir = cfg.experiments_dir / experiment_name

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    logger.info(f"Running diagnostics for experiment: {experiment_name}")

    # Load prices if available (for regime metrics)
    prices = None
    try:
        prices = load_price_parquet(
            allow_fallback=True,
            start_date=getattr(cfg, "price_start", None),
            end_date=getattr(cfg, "price_end", None),
            cfg=cfg,
        )
        if not prices.empty:
            logger.info(f"Loaded price data: {len(prices)} rows")
        else:
            logger.warning("Price data is empty, regime metrics will be skipped")
    except Exception as e:
        logger.warning(f"Could not load price data: {e}. Regime metrics will be skipped.")

    # Create diagnostics config
    risk_cfg = RiskFilterConfig()
    if exp_cfg:
        risk_filters = exp_cfg.get("risk_filters", {})
        if risk_filters:
            risk_cfg = RiskFilterConfig(**{k: v for k, v in risk_filters.items() if k in RiskFilterConfig.__annotations__})

    diag_cfg = DiagnosticsConfig(
        regime_vol_buckets=args.vol_buckets,
        risk_cfg=risk_cfg,
    )

    # Run diagnostics
    runner = DiagnosticsRunner(exp_dir, cfg=diag_cfg)
    results = runner.run(prices=prices, save_plots=not args.no_plots)

    # Print summary
    print("\n" + "=" * 80)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 80)

    if not results["yearly"].empty:
        print("\nPer-Year Metrics:")
        print(results["yearly"].to_string(index=False))
        print(f"\nSaved to: {exp_dir / 'diagnostics_yearly.csv'}")

    if not results["regime"].empty:
        print("\nRegime-Based Metrics:")
        print(results["regime"].to_string(index=False))
        print(f"\nSaved to: {exp_dir / 'diagnostics_regime.csv'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

