"""Core diagnostics computation for year-by-year and regime-based analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from intentflow_ai.backtest.filters import RiskFilterConfig, compute_regime_flags
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics computation."""

    regime_vol_buckets: List[float] = None  # e.g., [0, 33, 67, 100] for low/mid/high vol
    risk_cfg: RiskFilterConfig = None
    annualization_factor: float = 252.0  # Trading days per year

    def __post_init__(self):
        if self.regime_vol_buckets is None:
            self.regime_vol_buckets = [0, 33, 67, 100]  # Low, Mid, High
        if self.risk_cfg is None:
            self.risk_cfg = RiskFilterConfig()


def compute_per_year_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    *,
    date_col: str = "date_in",
    ret_col: str = "net_ret",
    equity_date_col: str = "date",
    annualization_factor: float = 252.0,
) -> pd.DataFrame:
    """Compute per-year performance metrics.

    Args:
        trades: DataFrame with trade-level returns
        equity: Series with equity curve indexed by date
        date_col: Column name for trade entry date
        ret_col: Column name for trade returns
        equity_date_col: Column name for equity curve dates
        annualization_factor: Trading days per year for annualization

    Returns:
        DataFrame with columns: year, CAGR, Sharpe, maxDD, win_rate, trade_count, avg_return
    """
    if trades.empty:
        return pd.DataFrame(
            columns=["year", "CAGR", "Sharpe", "maxDD", "win_rate", "trade_count", "avg_return"]
        )

    trades = trades.copy()
    trades[date_col] = pd.to_datetime(trades[date_col])
    equity = equity.copy()
    if isinstance(equity, pd.Series):
        equity.index = pd.to_datetime(equity.index)
    else:
        equity[equity_date_col] = pd.to_datetime(equity[equity_date_col])
        equity = equity.set_index(equity_date_col).iloc[:, 0]

    years = sorted(trades[date_col].dt.year.unique())
    results = []

    for year in years:
        year_trades = trades[trades[date_col].dt.year == year]
        if year_trades.empty:
            continue

        # Filter equity curve for this year
        year_equity = equity[equity.index.year == year]
        if year_equity.empty:
            continue

        # Compute metrics
        trade_count = len(year_trades)
        win_rate = float((year_trades[ret_col] > 0).mean())
        avg_return = float(year_trades[ret_col].mean())

        # Equity curve metrics
        if len(year_equity) > 1:
            daily_returns = year_equity.pct_change().dropna()
            if len(daily_returns) > 0:
                # CAGR
                start_val = year_equity.iloc[0]
                end_val = year_equity.iloc[-1]
                trading_days = len(year_equity)
                if start_val > 0 and trading_days > 0:
                    cagr = float((end_val / start_val) ** (annualization_factor / trading_days) - 1.0)
                else:
                    cagr = 0.0

                # Sharpe
                mean_ret = float(daily_returns.mean())
                std_ret = float(daily_returns.std())
                sharpe = float(mean_ret / (std_ret + 1e-12) * np.sqrt(annualization_factor)) if std_ret > 0 else 0.0

                # Max drawdown
                roll_max = year_equity.cummax()
                dd = (year_equity / roll_max) - 1.0
                maxdd = float(dd.min())
            else:
                cagr = 0.0
                sharpe = 0.0
                maxdd = 0.0
        else:
            cagr = 0.0
            sharpe = 0.0
            maxdd = 0.0

        results.append(
            {
                "year": int(year),
                "CAGR": cagr,
                "Sharpe": sharpe,
                "maxDD": maxdd,
                "win_rate": win_rate,
                "trade_count": trade_count,
                "avg_return": avg_return,
            }
        )

    return pd.DataFrame(results)


def compute_regime_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    prices: pd.DataFrame,
    *,
    date_col: str = "date_in",
    ret_col: str = "net_ret",
    equity_date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    cfg: DiagnosticsConfig = None,
) -> pd.DataFrame:
    """Compute performance metrics by market regime.

    Args:
        trades: DataFrame with trade-level returns
        equity: Series with equity curve indexed by date
        prices: DataFrame with price panel (date, ticker, close)
        date_col: Column name for trade entry date
        ret_col: Column name for trade returns
        equity_date_col: Column name for equity curve dates
        ticker_col: Column name for ticker in prices
        close_col: Column name for close price in prices
        cfg: Diagnostics configuration

    Returns:
        DataFrame with regime-based metrics
    """
    if cfg is None:
        cfg = DiagnosticsConfig()

    if trades.empty or prices.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades[date_col] = pd.to_datetime(trades[date_col])

    # Compute regime flags from price panel
    # Ensure prices has the right columns
    if "date" not in prices.columns:
        raise ValueError(f"prices DataFrame must have 'date' column")
    if ticker_col not in prices.columns:
        raise ValueError(f"prices DataFrame must have '{ticker_col}' column")
    if close_col not in prices.columns:
        raise ValueError(f"prices DataFrame must have '{close_col}' column")

    px_pivot = prices.pivot_table(index="date", columns=ticker_col, values=close_col)
    if px_pivot.empty:
        logger.warning("Empty price pivot, cannot compute regime metrics")
        return pd.DataFrame()

    px_pivot.index = pd.to_datetime(px_pivot.index)
    regime_flags = compute_regime_flags(px_pivot, cfg.risk_cfg)

    # Regime flags is a DataFrame indexed by date - convert to column
    regime_flags = regime_flags.reset_index()
    # The index becomes a column, rename it to "date" if needed
    if "index" in regime_flags.columns:
        regime_flags.rename(columns={"index": "date"}, inplace=True)
    elif regime_flags.index.name:
        regime_flags.rename_axis("date", inplace=True)
        regime_flags = regime_flags.reset_index()
    # Ensure date column exists and is datetime
    if "date" not in regime_flags.columns:
        regime_flags["date"] = pd.to_datetime(regime_flags.index)
    else:
        regime_flags["date"] = pd.to_datetime(regime_flags["date"])

    # Determine bull/bear from trend_ok
    regime_flags["regime_is_bull"] = regime_flags["trend_ok"].astype(int)
    regime_flags["regime_is_bear"] = (~regime_flags["trend_ok"]).astype(int)

    # Create volatility buckets from index_vol
    vol_buckets = cfg.regime_vol_buckets
    if len(vol_buckets) < 2:
        vol_buckets = [0, 33, 67, 100]

    # Compute volatility percentile for each date (expanding window)
    vol_series = regime_flags["index_vol"]
    vol_pct = vol_series.expanding(min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= 20 else np.nan, raw=False
    )
    regime_flags["index_vol_pct"] = vol_pct.values

    # Classify volatility regime
    def classify_vol_pct(pct):
        if pd.isna(pct):
            return "unknown"
        for i in range(len(vol_buckets) - 1):
            if vol_buckets[i] <= pct < vol_buckets[i + 1]:
                if i == 0:
                    return "low_vol"
                elif i == 1:
                    return "mid_vol"
                else:
                    return "high_vol"
        return "high_vol"

    regime_flags["vol_regime"] = regime_flags["index_vol_pct"].apply(classify_vol_pct)

    # Merge with trades
    trades_with_regime = trades.merge(
        regime_flags[["date", "regime_is_bull", "regime_is_bear", "vol_regime", "index_vol_pct"]],
        left_on=date_col,
        right_on="date",
        how="left",
    )

    results = []

    # By bull/bear
    for regime_type in ["bull", "bear"]:
        col = f"regime_is_{regime_type}"
        regime_trades = trades_with_regime[trades_with_regime[col] == 1]
        if not regime_trades.empty:
            results.append(
                {
                    "regime_type": regime_type,
                    "regime_subtype": "all",
                    "trade_count": len(regime_trades),
                    "win_rate": float((regime_trades[ret_col] > 0).mean()),
                    "avg_return": float(regime_trades[ret_col].mean()),
                    "total_return": float((1 + regime_trades[ret_col]).prod() - 1),
                }
            )

    # By volatility regime
    for vol_regime in ["low_vol", "mid_vol", "high_vol"]:
        vol_trades = trades_with_regime[trades_with_regime["vol_regime"] == vol_regime]
        if not vol_trades.empty:
            results.append(
                {
                    "regime_type": "volatility",
                    "regime_subtype": vol_regime,
                    "trade_count": len(vol_trades),
                    "win_rate": float((vol_trades[ret_col] > 0).mean()),
                    "avg_return": float(vol_trades[ret_col].mean()),
                    "total_return": float((1 + vol_trades[ret_col]).prod() - 1),
                }
            )

    # Combined: bull/bear x vol
    for trend in ["bull", "bear"]:
        for vol in ["low_vol", "mid_vol", "high_vol"]:
            trend_col = f"regime_is_{trend}"
            combined_trades = trades_with_regime[
                (trades_with_regime[trend_col] == 1) & (trades_with_regime["vol_regime"] == vol)
            ]
            if not combined_trades.empty:
                results.append(
                    {
                        "regime_type": f"{trend}_{vol}",
                        "regime_subtype": f"{trend}_{vol}",
                        "trade_count": len(combined_trades),
                        "win_rate": float((combined_trades[ret_col] > 0).mean()),
                        "avg_return": float(combined_trades[ret_col].mean()),
                        "total_return": float((1 + combined_trades[ret_col]).prod() - 1),
                    }
                )

    return pd.DataFrame(results)


@dataclass
class DiagnosticsRunner:
    """Main runner for diagnostics computation."""

    experiment_dir: Path
    cfg: DiagnosticsConfig = None

    def __init__(self, experiment_dir: Path | str, cfg: DiagnosticsConfig = None):
        self.experiment_dir = Path(experiment_dir)
        if cfg is None:
            cfg = DiagnosticsConfig()
        self.cfg = cfg

    def run(
        self,
        *,
        prices: Optional[pd.DataFrame] = None,
        save_plots: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Run all diagnostics and save results.

        Args:
            prices: Optional price panel. If None, will try to load from standard location.
            save_plots: Whether to save diagnostic plots

        Returns:
            Dictionary with 'yearly' and 'regime' DataFrames
        """
        # Load backtest outputs
        trades_path = self.experiment_dir / "bt_trades.csv"
        equity_path = self.experiment_dir / "bt_equity.csv"

        if not trades_path.exists():
            # Try alternative names
            trades_path = self.experiment_dir / "trades.csv"
        if not equity_path.exists():
            equity_path = self.experiment_dir / "equity_curve.csv"

        if not trades_path.exists():
            raise FileNotFoundError(f"Trades file not found in {self.experiment_dir}")
        if not equity_path.exists():
            raise FileNotFoundError(f"Equity curve file not found in {self.experiment_dir}")

        trades = pd.read_csv(trades_path, parse_dates=["date_in"])
        equity = pd.read_csv(equity_path, parse_dates=["date"])

        # Compute per-year metrics
        logger.info("Computing per-year metrics...")
        yearly_metrics = compute_per_year_metrics(
            trades,
            equity.set_index("date")["equity"],
            annualization_factor=self.cfg.annualization_factor,
        )

        # Compute regime metrics (need prices)
        regime_metrics = pd.DataFrame()
        if prices is not None:
            logger.info("Computing regime-based metrics...")
            regime_metrics = compute_regime_metrics(
                trades,
                equity.set_index("date")["equity"],
                prices,
                cfg=self.cfg,
            )
        else:
            logger.warning("No prices provided, skipping regime metrics")

        # Save results
        yearly_path = self.experiment_dir / "diagnostics_yearly.csv"
        yearly_json_path = self.experiment_dir / "diagnostics_yearly.json"
        regime_path = self.experiment_dir / "diagnostics_regime.csv"
        regime_json_path = self.experiment_dir / "diagnostics_regime.json"

        yearly_metrics.to_csv(yearly_path, index=False)
        yearly_metrics.to_json(yearly_json_path, orient="records", indent=2)

        if not regime_metrics.empty:
            regime_metrics.to_csv(regime_path, index=False)
            regime_metrics.to_json(regime_json_path, orient="records", indent=2)

        logger.info(f"Saved yearly diagnostics to {yearly_path}")
        if not regime_metrics.empty:
            logger.info(f"Saved regime diagnostics to {regime_path}")

        # Optional plots
        if save_plots:
            self._save_plots(yearly_metrics, equity, trades)

        return {"yearly": yearly_metrics, "regime": regime_metrics}

    def _save_plots(self, yearly_metrics: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
        """Save diagnostic plots."""
        try:
            import matplotlib.pyplot as plt

            plots_dir = self.experiment_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Plot 1: Equity curve by year
            if not yearly_metrics.empty and "year" in yearly_metrics.columns:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle("Per-Year Diagnostics", fontsize=16)

                # CAGR by year
                axes[0, 0].bar(yearly_metrics["year"], yearly_metrics["CAGR"])
                axes[0, 0].set_title("CAGR by Year")
                axes[0, 0].set_xlabel("Year")
                axes[0, 0].set_ylabel("CAGR")
                axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)

                # Sharpe by year
                axes[0, 1].bar(yearly_metrics["year"], yearly_metrics["Sharpe"])
                axes[0, 1].set_title("Sharpe Ratio by Year")
                axes[0, 1].set_xlabel("Year")
                axes[0, 1].set_ylabel("Sharpe")
                axes[0, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)

                # Win rate by year
                axes[1, 0].bar(yearly_metrics["year"], yearly_metrics["win_rate"])
                axes[1, 0].set_title("Win Rate by Year")
                axes[1, 0].set_xlabel("Year")
                axes[1, 0].set_ylabel("Win Rate")
                axes[1, 0].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

                # Trade count by year
                axes[1, 1].bar(yearly_metrics["year"], yearly_metrics["trade_count"])
                axes[1, 1].set_title("Trade Count by Year")
                axes[1, 1].set_xlabel("Year")
                axes[1, 1].set_ylabel("Number of Trades")

                plt.tight_layout()
                plt.savefig(plots_dir / "diagnostics_yearly.png", dpi=150, bbox_inches="tight")
                plt.close()

                logger.info(f"Saved yearly diagnostics plot to {plots_dir / 'diagnostics_yearly.png'}")

        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
        except Exception as e:
            logger.warning(f"Failed to save plots: {e}")

