#!/usr/bin/env python3
"""
Survivorship Bias Audit

Checks for survivorship bias in the training data:
1. Compares current universe to historical NIFTY constituents
2. Identifies delisted stocks (mergers, bankruptcies)
3. Flags tickers that appear/disappear mid-training period
4. Generates audit report

Survivorship bias occurs when we only train on stocks that survived,
leading to overly optimistic backtests.

Usage:
    python scripts/audit_survivorship.py --output reports/survivorship_audit.md
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


# Known delisted/merged stocks (incomplete list - should be expanded)
KNOWN_DELISTINGS = {
    "HDFC": {"delisted": "2023-07-13", "reason": "Merged with HDFC Bank"},
    "RCOM": {"delisted": "2020-12-17", "reason": "Bankruptcy"},
    "DHFL": {"delisted": "2021-06-14", "reason": "Resolution"},
    "YESBANK": {"delisted": None, "reason": "Still trading but restructured"},
    "RELCAPITAL": {"delisted": "2022-02-28", "reason": "Resolution"},
    "INFRATEL": {"delisted": "2020-11-19", "reason": "Merged with Bharti Airtel"},
    "IBULHSGFIN": {"delisted": "2023-04-28", "reason": "Restructured"},
}

# Known ticker changes/name changes
TICKER_CHANGES = {
    "HDFCLIFE": {"old": "HDFCSTDLIFE"},
    "SBILIFE": {"old": "SBILIFE"},
    "ICICIPRULI": {"old": "ICICIPRULI"},
    "LTIM": {"old": "MINDTREE"},
    "JIOFIN": {"old": "JFSL"},
}


def load_price_data(
    price_file: Optional[str] = None,
) -> pd.DataFrame:
    """Load price data for survivorship analysis."""
    if price_file:
        path = Path(price_file)
    else:
        path = settings.data_dir / "raw" / "price_confirmation" / "all_prices.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_universe(
    universe_file: Optional[str] = None,
) -> Set[str]:
    """Load current universe tickers."""
    if universe_file:
        path = Path(universe_file)
    else:
        path = settings.data_dir / "static" / "sector_map.csv"
    
    if not path.exists():
        return set()
    
    df = pd.read_csv(path)
    
    if "ticker_nse" in df.columns:
        return set(df["ticker_nse"].dropna())
    elif "ticker" in df.columns:
        return set(df["ticker"].dropna())
    
    return set()


def analyze_ticker_coverage(
    price_df: pd.DataFrame,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Analyze when each ticker appears/disappears in the data.
    
    Returns:
        Dictionary mapping ticker -> {first_date, last_date, days_covered, gaps}
    """
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else price_df["date"].max()
    
    price_df = price_df[(price_df["date"] >= start) & (price_df["date"] <= end)]
    
    coverage = {}
    
    for ticker, group in price_df.groupby("ticker"):
        group = group.sort_values("date")
        
        first_date = group["date"].min()
        last_date = group["date"].max()
        days_covered = len(group)
        
        # Check for gaps > 30 days
        date_diffs = group["date"].diff()
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=30)]
        
        coverage[ticker] = {
            "first_date": first_date,
            "last_date": last_date,
            "days_covered": days_covered,
            "gaps": len(large_gaps),
            "max_gap_days": date_diffs.max().days if len(date_diffs) > 0 else 0,
        }
    
    return coverage


def find_disappeared_tickers(
    coverage: Dict[str, Dict],
    cutoff_date: str = "2024-01-01",
    min_coverage_days: int = 100,
) -> List[Tuple[str, str, int]]:
    """
    Find tickers that disappeared before cutoff date.
    
    Returns:
        List of (ticker, last_date, days_covered) tuples
    """
    cutoff = pd.to_datetime(cutoff_date)
    
    disappeared = []
    
    for ticker, info in coverage.items():
        if info["last_date"] < cutoff and info["days_covered"] >= min_coverage_days:
            disappeared.append((
                ticker,
                str(info["last_date"].date()),
                info["days_covered"],
            ))
    
    # Sort by last_date descending
    disappeared.sort(key=lambda x: x[1], reverse=True)
    
    return disappeared


def find_late_additions(
    coverage: Dict[str, Dict],
    train_start: str = "2015-01-01",
    threshold_date: str = "2020-01-01",
) -> List[Tuple[str, str, int]]:
    """
    Find tickers that only appear after a threshold date.
    
    These are potential survivorship bias sources as they
    may only be in the data because they performed well.
    """
    threshold = pd.to_datetime(threshold_date)
    train_start_dt = pd.to_datetime(train_start)
    
    late_additions = []
    
    for ticker, info in coverage.items():
        if info["first_date"] > threshold:
            days_late = (info["first_date"] - train_start_dt).days
            late_additions.append((
                ticker,
                str(info["first_date"].date()),
                days_late,
            ))
    
    # Sort by first_date
    late_additions.sort(key=lambda x: x[1])
    
    return late_additions


def check_known_delistings(
    coverage: Dict[str, Dict],
    current_universe: Set[str],
) -> List[Dict]:
    """Check coverage of known delisted stocks."""
    results = []
    
    for ticker, info in KNOWN_DELISTINGS.items():
        in_coverage = ticker in coverage
        in_universe = ticker in current_universe
        
        results.append({
            "ticker": ticker,
            "reason": info["reason"],
            "delisted_date": info["delisted"],
            "in_price_data": in_coverage,
            "in_current_universe": in_universe,
            "last_price_date": str(coverage[ticker]["last_date"].date()) if in_coverage else None,
        })
    
    return results


def calculate_survivorship_risk_score(
    coverage: Dict[str, Dict],
    current_universe: Set[str],
    train_start: str,
    train_end: str,
) -> Dict:
    """
    Calculate overall survivorship bias risk score.
    
    Higher score = higher risk of bias.
    """
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    train_days = (train_end_dt - train_start_dt).days
    
    # Metrics
    total_tickers = len(coverage)
    
    # 1. Tickers that don't cover full period
    incomplete_coverage = sum(
        1 for t, info in coverage.items()
        if (info["last_date"] - info["first_date"]).days < train_days * 0.8
    )
    incomplete_pct = incomplete_coverage / total_tickers if total_tickers > 0 else 0
    
    # 2. Late additions (after 30% of training period)
    late_threshold = train_start_dt + pd.Timedelta(days=train_days * 0.3)
    late_additions = sum(
        1 for t, info in coverage.items()
        if info["first_date"] > late_threshold
    )
    late_pct = late_additions / total_tickers if total_tickers > 0 else 0
    
    # 3. Early dropouts (before 70% of training period)
    early_cutoff = train_start_dt + pd.Timedelta(days=train_days * 0.7)
    early_dropouts = sum(
        1 for t, info in coverage.items()
        if info["last_date"] < early_cutoff
    )
    dropout_pct = early_dropouts / total_tickers if total_tickers > 0 else 0
    
    # 4. Universe mismatch
    in_price_not_universe = len(set(coverage.keys()) - current_universe)
    in_universe_not_price = len(current_universe - set(coverage.keys()))
    mismatch_pct = (in_price_not_universe + in_universe_not_price) / (total_tickers + 1)
    
    # Composite risk score (0-100)
    risk_score = (
        incomplete_pct * 25 +
        late_pct * 30 +
        dropout_pct * 25 +
        mismatch_pct * 20
    ) * 100
    
    return {
        "risk_score": risk_score,
        "risk_level": "HIGH" if risk_score > 30 else "MEDIUM" if risk_score > 15 else "LOW",
        "total_tickers": total_tickers,
        "incomplete_coverage_pct": incomplete_pct * 100,
        "late_additions_pct": late_pct * 100,
        "early_dropouts_pct": dropout_pct * 100,
        "universe_mismatch_pct": mismatch_pct * 100,
    }


def generate_audit_report(
    coverage: Dict[str, Dict],
    current_universe: Set[str],
    train_start: str,
    train_end: str,
    output_path: Optional[str] = None,
) -> str:
    """Generate survivorship bias audit report."""
    
    risk = calculate_survivorship_risk_score(
        coverage, current_universe, train_start, train_end
    )
    
    disappeared = find_disappeared_tickers(coverage, train_end)
    late_adds = find_late_additions(coverage, train_start)
    known_delists = check_known_delistings(coverage, current_universe)
    
    lines = [
        "# Survivorship Bias Audit Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Training Period:** {train_start} to {train_end}",
        "",
        "---",
        "",
        "## Risk Assessment",
        "",
        f"**Overall Risk Score:** {risk['risk_score']:.1f}/100 ({risk['risk_level']})",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Tickers | {risk['total_tickers']} |",
        f"| Incomplete Coverage | {risk['incomplete_coverage_pct']:.1f}% |",
        f"| Late Additions | {risk['late_additions_pct']:.1f}% |",
        f"| Early Dropouts | {risk['early_dropouts_pct']:.1f}% |",
        f"| Universe Mismatch | {risk['universe_mismatch_pct']:.1f}% |",
        "",
        "---",
        "",
        "## Disappeared Tickers",
        "",
        "Tickers that stopped trading before the end of training period:",
        "",
        "| Ticker | Last Date | Days Covered |",
        "|--------|-----------|--------------|",
    ]
    
    for ticker, last_date, days in disappeared[:20]:
        lines.append(f"| {ticker} | {last_date} | {days} |")
    
    if len(disappeared) > 20:
        lines.append(f"| ... | ... | ({len(disappeared) - 20} more) |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Late Additions",
        "",
        "Tickers added after significant portion of training period:",
        "",
        "| Ticker | First Date | Days Late |",
        "|--------|------------|-----------|",
    ])
    
    for ticker, first_date, days_late in late_adds[:20]:
        lines.append(f"| {ticker} | {first_date} | {days_late} |")
    
    if len(late_adds) > 20:
        lines.append(f"| ... | ... | ({len(late_adds) - 20} more) |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Known Delisted Stocks",
        "",
        "| Ticker | Reason | In Price Data | In Universe |",
        "|--------|--------|---------------|-------------|",
    ])
    
    for info in known_delists:
        lines.append(
            f"| {info['ticker']} | {info['reason']} | "
            f"{'Yes' if info['in_price_data'] else 'No'} | "
            f"{'Yes' if info['in_current_universe'] else 'No'} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])
    
    if risk['risk_level'] == "HIGH":
        lines.extend([
            "⚠️ **HIGH RISK** - Significant survivorship bias potential",
            "",
            "1. Consider using point-in-time constituent data",
            "2. Include delisted stocks in training data",
            "3. Be cautious with backtest results",
            "4. Apply haircut to expected returns",
        ])
    elif risk['risk_level'] == "MEDIUM":
        lines.extend([
            "⚡ **MEDIUM RISK** - Some survivorship bias possible",
            "",
            "1. Review late additions for selection bias",
            "2. Validate results on out-of-sample data",
            "3. Consider including more historical constituents",
        ])
    else:
        lines.extend([
            "✅ **LOW RISK** - Survivorship bias appears manageable",
            "",
            "1. Continue monitoring for new delistings",
            "2. Periodically update historical constituent data",
        ])
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)
        logger.info(f"Saved audit report to {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit for survivorship bias")
    parser.add_argument(
        "--price-file",
        default=None,
        help="Path to price data file",
    )
    parser.add_argument(
        "--universe-file",
        default=None,
        help="Path to universe file",
    )
    parser.add_argument(
        "--train-start",
        default="2015-01-01",
        help="Training period start date",
    )
    parser.add_argument(
        "--train-end",
        default="2024-12-31",
        help="Training period end date",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (defaults to stdout)",
    )
    
    args = parser.parse_args()
    
    print("Loading price data...", file=sys.stderr)
    price_df = load_price_data(args.price_file)
    
    print("Loading universe...", file=sys.stderr)
    universe = load_universe(args.universe_file)
    
    print("Analyzing ticker coverage...", file=sys.stderr)
    coverage = analyze_ticker_coverage(
        price_df,
        start_date=args.train_start,
        end_date=args.train_end,
    )
    
    print("Generating report...", file=sys.stderr)
    report = generate_audit_report(
        coverage=coverage,
        current_universe=universe,
        train_start=args.train_start,
        train_end=args.train_end,
        output_path=args.output,
    )
    
    if not args.output:
        print(report)
    
    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
