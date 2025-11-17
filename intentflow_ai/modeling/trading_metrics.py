"""Comprehensive trading metrics for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradingMetrics:
    """Comprehensive trading metrics."""

    # Tier 1: Decision Metrics
    expected_value: float  # Average profit per trade
    profit_factor: float  # Sum of wins / Sum of losses
    hit_rate: float  # % of profitable trades

    # Tier 2: Quality Metrics
    return_ic: float  # Pearson correlation of signal magnitude with returns
    contribution_ic: float  # Correlation of (signal × position_size) with returns
    rank_ic: float  # Spearman correlation (existing)
    decile_ic: float  # (Top decile return - Bottom decile return) / Avg Vol
    sharpe_by_decile: List[float]  # Sharpe ratio for each decile

    # Tier 3: Risk Metrics
    max_drawdown: float
    calmar_ratio: float  # Return / Max DD
    sortino_ratio: float  # Return / Downside volatility

    # Additional
    avg_win: float
    avg_loss: float
    win_count: int
    loss_count: int
    total_trades: int


def compute_hit_rate(trades_df: pd.DataFrame, ret_col: str = "net_ret") -> float:
    """Compute hit rate (% of profitable trades).

    Args:
        trades_df: DataFrame with trade returns
        ret_col: Column name for returns

    Returns:
        Hit rate as float (0-1)
    """
    if trades_df.empty or ret_col not in trades_df.columns:
        return 0.0
    return float((trades_df[ret_col] > 0).mean())


def compute_profit_factor(trades_df: pd.DataFrame, ret_col: str = "net_ret") -> float:
    """Compute profit factor (Sum of wins / Sum of losses).

    Args:
        trades_df: DataFrame with trade returns
        ret_col: Column name for returns

    Returns:
        Profit factor (inf if no losses)
    """
    if trades_df.empty or ret_col not in trades_df.columns:
        return 0.0

    wins = trades_df[trades_df[ret_col] > 0][ret_col].sum()
    losses = abs(trades_df[trades_df[ret_col] < 0][ret_col].sum())

    if losses == 0:
        return float("inf") if wins > 0 else 0.0

    return float(wins / losses)


def compute_expected_value(
    trades_df: pd.DataFrame,
    ret_col: str = "net_ret",
) -> Tuple[float, Dict[str, float]]:
    """Compute expected value per trade.

    Args:
        trades_df: DataFrame with trade returns
        ret_col: Column name for returns

    Returns:
        Tuple of (expected_value, breakdown_dict)
    """
    if trades_df.empty or ret_col not in trades_df.columns:
        return 0.0, {}

    wins = trades_df[trades_df[ret_col] > 0]
    losses = trades_df[trades_df[ret_col] < 0]

    hit_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0
    loss_rate = len(losses) / len(trades_df) if len(trades_df) > 0 else 0.0

    avg_win = float(wins[ret_col].mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses[ret_col].mean())) if len(losses) > 0 else 0.0

    ev = (hit_rate * avg_win) - (loss_rate * avg_loss)

    breakdown = {
        "hit_rate": hit_rate,
        "loss_rate": loss_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_count": len(wins),
        "loss_count": len(losses),
        "total_trades": len(trades_df),
    }

    return float(ev), breakdown


def compute_return_ic(
    signal_strength: pd.Series,
    actual_returns: pd.Series,
) -> float:
    """Compute Return IC (Pearson correlation of signal magnitude with returns).

    Args:
        signal_strength: Signal probabilities or scores
        actual_returns: Actual returns (aligned with signals)

    Returns:
        Return IC (Pearson correlation)
    """
    aligned = pd.DataFrame({"signal": signal_strength, "return": actual_returns}).dropna()
    if len(aligned) < 10:
        return np.nan

    try:
        return float(aligned["signal"].corr(aligned["return"], method="pearson"))
    except Exception:
        return np.nan


def compute_contribution_ic(
    signal_strength: pd.Series,
    position_size: pd.Series,
    actual_returns: pd.Series,
) -> float:
    """Compute Contribution-weighted IC.

    Args:
        signal_strength: Signal probabilities or scores
        position_size: Position sizes (capital allocation)
        actual_returns: Actual returns

    Returns:
        Contribution IC (correlation of signal × position_size with returns)
    """
    aligned = pd.DataFrame(
        {
            "signal": signal_strength,
            "position_size": position_size,
            "return": actual_returns,
        }
    ).dropna()

    if len(aligned) < 10:
        return np.nan

    # Contribution = signal strength × position size
    contribution = aligned["signal"] * aligned["position_size"]

    try:
        return float(contribution.corr(aligned["return"], method="pearson"))
    except Exception:
        return np.nan


def compute_decile_ic(
    signal_strength: pd.Series,
    actual_returns: pd.Series,
    *,
    n_deciles: int = 10,
) -> Tuple[float, pd.DataFrame]:
    """Compute Decile IC analysis.

    Args:
        signal_strength: Signal probabilities or scores
        actual_returns: Actual returns
        n_deciles: Number of deciles (default: 10)

    Returns:
        Tuple of (decile_ic, decile_stats DataFrame)
    """
    aligned = pd.DataFrame({"signal": signal_strength, "return": actual_returns}).dropna()

    if len(aligned) < n_deciles:
        return np.nan, pd.DataFrame()

    # Split into deciles
    aligned["decile"] = pd.qcut(
        aligned["signal"],
        q=n_deciles,
        labels=False,
        duplicates="drop",
    )

    decile_stats = []
    decile_returns = []

    for d in range(n_deciles):
        decile_data = aligned[aligned["decile"] == d]
        if len(decile_data) == 0:
            continue

        decile_return = float(decile_data["return"].mean())
        decile_std = float(decile_data["return"].std())
        decile_sharpe = decile_return / (decile_std + 1e-9) if decile_std > 0 else 0.0
        decile_count = len(decile_data)

        decile_stats.append(
            {
                "decile": d,
                "mean_return": decile_return,
                "std_return": decile_std,
                "sharpe": decile_sharpe,
                "count": decile_count,
                "min_signal": float(decile_data["signal"].min()),
                "max_signal": float(decile_data["signal"].max()),
            }
        )
        decile_returns.append(decile_return)

    decile_df = pd.DataFrame(decile_stats)

    # Decile IC = (Top decile return - Bottom decile return) / Avg Vol
    if len(decile_returns) >= 2:
        top_return = decile_returns[-1]  # Highest decile
        bottom_return = decile_returns[0]  # Lowest decile
        avg_vol = float(aligned["return"].std())
        decile_ic = (top_return - bottom_return) / (avg_vol + 1e-9)
    else:
        decile_ic = np.nan

    return float(decile_ic), decile_df


def compute_sharpe_by_decile(
    signal_strength: pd.Series,
    actual_returns: pd.Series,
    *,
    n_deciles: int = 10,
    annualization_factor: float = 252.0,
) -> List[float]:
    """Compute Sharpe ratio for each decile.

    Args:
        signal_strength: Signal probabilities or scores
        actual_returns: Actual returns
        n_deciles: Number of deciles
        annualization_factor: Trading days per year

    Returns:
        List of Sharpe ratios (one per decile)
    """
    aligned = pd.DataFrame({"signal": signal_strength, "return": actual_returns}).dropna()

    if len(aligned) < n_deciles:
        return [np.nan] * n_deciles

    aligned["decile"] = pd.qcut(
        aligned["signal"],
        q=n_deciles,
        labels=False,
        duplicates="drop",
    )

    sharpes = []
    for d in range(n_deciles):
        decile_data = aligned[aligned["decile"] == d]
        if len(decile_data) == 0:
            sharpes.append(np.nan)
            continue

        mean_ret = float(decile_data["return"].mean())
        std_ret = float(decile_data["return"].std())
        sharpe = (
            float(mean_ret / (std_ret + 1e-9) * np.sqrt(annualization_factor))
            if std_ret > 0
            else 0.0
        )
        sharpes.append(sharpe)

    return sharpes


def compute_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """Compute Calmar ratio (Return / Max Drawdown).

    Args:
        total_return: Total return (CAGR or cumulative)
        max_drawdown: Maximum drawdown (negative value)

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float("inf") if total_return > 0 else 0.0
    return float(total_return / abs(max_drawdown))


def compute_sortino_ratio(
    returns: pd.Series,
    *,
    annualization_factor: float = 252.0,
    target_return: float = 0.0,
) -> float:
    """Compute Sortino ratio (penalizes only downside volatility).

    Args:
        returns: Series of returns
        annualization_factor: Trading days per year
        target_return: Target return (default: 0)

    Returns:
        Sortino ratio
    """
    if returns.empty:
        return 0.0

    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float("inf") if returns.mean() > 0 else 0.0

    downside_std = float(downside_returns.std())
    mean_return = float(returns.mean())

    if downside_std == 0:
        return float("inf") if mean_return > 0 else 0.0

    sortino = mean_return / downside_std * np.sqrt(annualization_factor)
    return float(sortino)


def compute_comprehensive_trading_metrics(
    trades_df: pd.DataFrame,
    *,
    ret_col: str = "net_ret",
    signal_strength: Optional[pd.Series] = None,
    actual_returns: Optional[pd.Series] = None,
    position_sizes: Optional[pd.Series] = None,
    equity_curve: Optional[pd.Series] = None,
    total_return: Optional[float] = None,
) -> TradingMetrics:
    """Compute all comprehensive trading metrics.

    Args:
        trades_df: DataFrame with trades
        ret_col: Column name for returns in trades_df
        signal_strength: Optional signal probabilities (for IC metrics)
        actual_returns: Optional actual returns (for IC metrics)
        position_sizes: Optional position sizes (for contribution IC)
        equity_curve: Optional equity curve (for drawdown)
        total_return: Optional total return (for Calmar)

    Returns:
        TradingMetrics dataclass
    """
    # Tier 1: Decision Metrics
    hit_rate = compute_hit_rate(trades_df, ret_col)
    profit_factor = compute_profit_factor(trades_df, ret_col)
    ev, ev_breakdown = compute_expected_value(trades_df, ret_col)

    # Tier 2: Quality Metrics
    return_ic = np.nan
    contribution_ic = np.nan
    rank_ic = np.nan
    decile_ic = np.nan
    sharpe_by_decile = []

    if signal_strength is not None and actual_returns is not None:
        return_ic = compute_return_ic(signal_strength, actual_returns)
        rank_ic = compute_return_ic(signal_strength, actual_returns)  # Will use spearman below

        # Rank IC (Spearman)
        aligned = pd.DataFrame({"signal": signal_strength, "return": actual_returns}).dropna()
        if len(aligned) >= 10:
            try:
                rank_ic = float(aligned["signal"].corr(aligned["return"], method="spearman"))
            except Exception:
                rank_ic = np.nan

        # Contribution IC
        if position_sizes is not None:
            contribution_ic = compute_contribution_ic(signal_strength, position_sizes, actual_returns)

        # Decile analysis
        decile_ic, _ = compute_decile_ic(signal_strength, actual_returns)
        sharpe_by_decile = compute_sharpe_by_decile(signal_strength, actual_returns)

    # Tier 3: Risk Metrics
    max_drawdown = 0.0
    calmar_ratio = 0.0
    sortino_ratio = 0.0

    if equity_curve is not None and not equity_curve.empty:
        roll_max = equity_curve.cummax()
        dd = (equity_curve / roll_max) - 1.0
        max_drawdown = float(dd.min()) if not dd.empty else 0.0

        # Compute returns from equity curve
        returns = equity_curve.pct_change().dropna()
        sortino_ratio = compute_sortino_ratio(returns)

    if total_return is not None:
        calmar_ratio = compute_calmar_ratio(total_return, max_drawdown)

    # Extract win/loss stats
    wins = trades_df[trades_df[ret_col] > 0]
    losses = trades_df[trades_df[ret_col] < 0]

    return TradingMetrics(
        expected_value=ev,
        profit_factor=profit_factor,
        hit_rate=hit_rate,
        return_ic=return_ic,
        contribution_ic=contribution_ic,
        rank_ic=rank_ic,
        decile_ic=decile_ic,
        sharpe_by_decile=sharpe_by_decile,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        sortino_ratio=sortino_ratio,
        avg_win=ev_breakdown.get("avg_win", 0.0),
        avg_loss=ev_breakdown.get("avg_loss", 0.0),
        win_count=ev_breakdown.get("win_count", 0),
        loss_count=ev_breakdown.get("loss_count", 0),
        total_trades=ev_breakdown.get("total_trades", 0),
    )


def compare_in_sample_vs_out_of_sample(
    train_signal: pd.Series,
    train_returns: pd.Series,
    test_signal: pd.Series,
    test_returns: pd.Series,
    *,
    train_position_sizes: Optional[pd.Series] = None,
    test_position_sizes: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compare IC metrics between in-sample and out-of-sample data.

    Args:
        train_signal: Training set signal strengths
        train_returns: Training set returns
        test_signal: Test set signal strengths
        test_returns: Test set returns
        train_position_sizes: Optional training position sizes
        test_position_sizes: Optional test position sizes

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    # Return IC
    train_return_ic = compute_return_ic(train_signal, train_returns)
    test_return_ic = compute_return_ic(test_signal, test_returns)

    # Rank IC
    train_aligned = pd.DataFrame({"signal": train_signal, "return": train_returns}).dropna()
    test_aligned = pd.DataFrame({"signal": test_signal, "return": test_returns}).dropna()

    train_rank_ic = (
        float(train_aligned["signal"].corr(train_aligned["return"], method="spearman"))
        if len(train_aligned) >= 10
        else np.nan
    )
    test_rank_ic = (
        float(test_aligned["signal"].corr(test_aligned["return"], method="spearman"))
        if len(test_aligned) >= 10
        else np.nan
    )

    # Contribution IC
    train_contribution_ic = np.nan
    test_contribution_ic = np.nan

    if train_position_sizes is not None:
        train_contribution_ic = compute_contribution_ic(train_signal, train_position_sizes, train_returns)
    if test_position_sizes is not None:
        test_contribution_ic = compute_contribution_ic(test_signal, test_position_sizes, test_returns)

    # Decile IC
    train_decile_ic, _ = compute_decile_ic(train_signal, train_returns)
    test_decile_ic, _ = compute_decile_ic(test_signal, test_returns)

    results.append(
        {
            "metric": "return_ic",
            "in_sample": train_return_ic,
            "out_of_sample": test_return_ic,
            "gap": train_return_ic - test_return_ic,
            "gap_pct": ((train_return_ic - test_return_ic) / (abs(train_return_ic) + 1e-9)) * 100,
        }
    )

    results.append(
        {
            "metric": "rank_ic",
            "in_sample": train_rank_ic,
            "out_of_sample": test_rank_ic,
            "gap": train_rank_ic - test_rank_ic,
            "gap_pct": ((train_rank_ic - test_rank_ic) / (abs(train_rank_ic) + 1e-9)) * 100,
        }
    )

    if not pd.isna(train_contribution_ic) or not pd.isna(test_contribution_ic):
        results.append(
            {
                "metric": "contribution_ic",
                "in_sample": train_contribution_ic,
                "out_of_sample": test_contribution_ic,
                "gap": train_contribution_ic - test_contribution_ic,
                "gap_pct": (
                    ((train_contribution_ic - test_contribution_ic) / (abs(train_contribution_ic) + 1e-9)) * 100
                    if not pd.isna(train_contribution_ic)
                    else np.nan
                ),
            }
        )

    results.append(
        {
            "metric": "decile_ic",
            "in_sample": train_decile_ic,
            "out_of_sample": test_decile_ic,
            "gap": train_decile_ic - test_decile_ic,
            "gap_pct": ((train_decile_ic - test_decile_ic) / (abs(train_decile_ic) + 1e-9)) * 100,
        }
    )

    return pd.DataFrame(results)

