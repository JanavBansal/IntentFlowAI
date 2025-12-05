#!/usr/bin/env python3
"""
EODHD Point-in-Time Audit

Verifies that EODHD fundamental data is correctly time-enforced:
1. Checks filing_date vs report_date alignment
2. Verifies available_date = report_date + reporting_delay
3. Sample checks against actual filing dates
4. Identifies potential lookahead bias

Lookahead bias is critical - using fundamental data before it was 
publicly available leads to unrealistically good backtests.

Usage:
    python scripts/audit_eodhd_pit.py --output reports/pit_audit.md
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


# Indian company reporting requirements:
# - Quarterly results: Within 45 days of quarter end
# - Annual results: Within 60 days of financial year end (March 31)
DEFAULT_REPORTING_DELAY = 45  # Days


def load_eodhd_fundamentals(
    parquet_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load parsed EODHD fundamentals."""
    if parquet_path:
        path = Path(parquet_path)
    else:
        path = settings.data_dir / "cache" / "fundamentals" / "eodhd_full.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"EODHD fundamentals not found: {path}")
    
    df = pd.read_parquet(path)
    return df


def load_raw_eodhd_json(
    ticker: str,
    eodhd_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Load raw EODHD JSON for a ticker."""
    if eodhd_dir:
        path = Path(eodhd_dir) / f"{ticker}.json"
    else:
        path = settings.data_dir / "raw" / "eodhd" / f"{ticker}.json"
    
    if not path.exists():
        return None
    
    with open(path, "r") as f:
        return json.load(f)


def check_filing_dates(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Check filing_date availability and quality.
    
    Returns:
        Dictionary with filing date statistics
    """
    total_records = len(df)
    
    # Check filing_date presence
    has_filing_date = df["filing_date"].notna().sum()
    filing_date_pct = has_filing_date / total_records * 100 if total_records > 0 else 0
    
    # Check report_date presence
    has_report_date = df["report_date"].notna().sum()
    
    # Check available_date presence
    has_available_date = df["available_date"].notna().sum()
    
    return {
        "total_records": total_records,
        "has_filing_date": has_filing_date,
        "filing_date_pct": filing_date_pct,
        "has_report_date": has_report_date,
        "has_available_date": has_available_date,
    }


def verify_reporting_delay(
    df: pd.DataFrame,
    expected_delay_days: int = DEFAULT_REPORTING_DELAY,
) -> Dict[str, Any]:
    """
    Verify that available_date = report_date + expected_delay.
    
    Returns:
        Dictionary with verification results
    """
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    
    # Calculate actual delay
    df["actual_delay"] = (df["available_date"] - df["report_date"]).dt.days
    
    # Check against expected
    df["correct_delay"] = df["actual_delay"] == expected_delay_days
    
    # For records with filing_date, check if filing_date is before available_date
    has_filing = df["filing_date"].notna()
    df.loc[has_filing, "filing_before_available"] = (
        df.loc[has_filing, "filing_date"] <= df.loc[has_filing, "available_date"]
    )
    
    return {
        "total_checked": len(df),
        "correct_delay_count": df["correct_delay"].sum(),
        "correct_delay_pct": df["correct_delay"].mean() * 100,
        "avg_actual_delay": df["actual_delay"].mean(),
        "min_delay": df["actual_delay"].min(),
        "max_delay": df["actual_delay"].max(),
        "filing_before_available_pct": df["filing_before_available"].mean() * 100 if has_filing.any() else None,
    }


def find_potential_lookahead(
    df: pd.DataFrame,
) -> List[Dict]:
    """
    Find records where data might be used before it was available.
    
    A record has potential lookahead if:
    - available_date is before or same as report_date (impossible)
    - filing_date is after available_date (data used before filed)
    """
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    
    issues = []
    
    # Check 1: available_date <= report_date (impossible in real world)
    impossible = df[df["available_date"] <= df["report_date"]]
    for _, row in impossible.iterrows():
        issues.append({
            "symbol": row["symbol"],
            "report_date": str(row["report_date"].date()),
            "available_date": str(row["available_date"].date()),
            "issue": "available_date <= report_date (impossible)",
            "severity": "HIGH",
        })
    
    # Check 2: filing_date > available_date (data used before it was filed)
    has_filing = df["filing_date"].notna()
    late_filing = df[has_filing & (df["filing_date"] > df["available_date"])]
    for _, row in late_filing.iterrows():
        issues.append({
            "symbol": row["symbol"],
            "report_date": str(row["report_date"].date()),
            "filing_date": str(row["filing_date"].date()),
            "available_date": str(row["available_date"].date()),
            "issue": "filing_date > available_date (lookahead)",
            "severity": "MEDIUM",
        })
    
    return issues


def sample_check_against_filings(
    df: pd.DataFrame,
    sample_size: int = 10,
) -> List[Dict]:
    """
    Sample check: Compare EODHD data against known filing dates.
    
    Note: This is a placeholder. In production, you'd compare against
    actual BSE/NSE filing dates from their websites.
    """
    # Sample random records
    if len(df) < sample_size:
        sample = df
    else:
        sample = df.sample(n=sample_size, random_state=42)
    
    results = []
    
    for _, row in sample.iterrows():
        result = {
            "symbol": row["symbol"],
            "report_date": str(row["report_date"]),
            "filing_date": str(row.get("filing_date", "N/A")),
            "available_date": str(row["available_date"]),
            "revenue": row.get("revenue"),
            "net_income": row.get("net_income"),
            "note": "Manual verification recommended",
        }
        results.append(result)
    
    return results


def calculate_pit_risk_score(
    filing_stats: Dict,
    delay_stats: Dict,
    lookahead_issues: List[Dict],
) -> Dict:
    """
    Calculate overall point-in-time data quality score.
    
    Higher score = better data quality (less lookahead risk).
    """
    score = 100.0
    issues = []
    
    # Penalty for missing filing dates
    if filing_stats["filing_date_pct"] < 50:
        penalty = (50 - filing_stats["filing_date_pct"]) * 0.5
        score -= penalty
        issues.append(f"Only {filing_stats['filing_date_pct']:.1f}% records have filing_date")
    
    # Penalty for incorrect delays
    if delay_stats["correct_delay_pct"] < 90:
        penalty = (90 - delay_stats["correct_delay_pct"]) * 0.3
        score -= penalty
        issues.append(f"Only {delay_stats['correct_delay_pct']:.1f}% records have correct delay")
    
    # Penalty for lookahead issues
    high_severity = sum(1 for i in lookahead_issues if i["severity"] == "HIGH")
    medium_severity = sum(1 for i in lookahead_issues if i["severity"] == "MEDIUM")
    
    if high_severity > 0:
        score -= high_severity * 5
        issues.append(f"{high_severity} high-severity lookahead issues")
    
    if medium_severity > 0:
        score -= medium_severity * 1
        issues.append(f"{medium_severity} medium-severity lookahead issues")
    
    score = max(0, min(100, score))
    
    if score >= 80:
        level = "GOOD"
    elif score >= 60:
        level = "ACCEPTABLE"
    elif score >= 40:
        level = "CAUTION"
    else:
        level = "POOR"
    
    return {
        "score": score,
        "level": level,
        "issues": issues,
    }


def generate_pit_audit_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """Generate point-in-time audit report."""
    
    filing_stats = check_filing_dates(df)
    delay_stats = verify_reporting_delay(df)
    lookahead_issues = find_potential_lookahead(df)
    sample_checks = sample_check_against_filings(df)
    risk = calculate_pit_risk_score(filing_stats, delay_stats, lookahead_issues)
    
    lines = [
        "# EODHD Point-in-Time Audit Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total Records:** {filing_stats['total_records']:,}",
        "",
        "---",
        "",
        "## Data Quality Score",
        "",
        f"**Score:** {risk['score']:.1f}/100 ({risk['level']})",
        "",
    ]
    
    if risk["issues"]:
        lines.append("**Issues Found:**")
        for issue in risk["issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Filing Date Coverage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Records | {filing_stats['total_records']:,} |",
        f"| With Filing Date | {filing_stats['has_filing_date']:,} ({filing_stats['filing_date_pct']:.1f}%) |",
        f"| With Report Date | {filing_stats['has_report_date']:,} |",
        f"| With Available Date | {filing_stats['has_available_date']:,} |",
        "",
        "---",
        "",
        "## Reporting Delay Verification",
        "",
        f"Expected delay: {DEFAULT_REPORTING_DELAY} days",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Records Checked | {delay_stats['total_checked']:,} |",
        f"| Correct Delay | {delay_stats['correct_delay_count']:,} ({delay_stats['correct_delay_pct']:.1f}%) |",
        f"| Average Delay | {delay_stats['avg_actual_delay']:.1f} days |",
        f"| Min Delay | {delay_stats['min_delay']} days |",
        f"| Max Delay | {delay_stats['max_delay']} days |",
    ])
    
    if delay_stats["filing_before_available_pct"] is not None:
        lines.append(f"| Filing Before Available | {delay_stats['filing_before_available_pct']:.1f}% |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Potential Lookahead Issues",
        "",
    ])
    
    if lookahead_issues:
        lines.extend([
            f"Found {len(lookahead_issues)} potential lookahead issues:",
            "",
            "| Symbol | Report Date | Issue | Severity |",
            "|--------|-------------|-------|----------|",
        ])
        
        for issue in lookahead_issues[:20]:
            lines.append(
                f"| {issue['symbol']} | {issue['report_date']} | "
                f"{issue['issue']} | {issue['severity']} |"
            )
        
        if len(lookahead_issues) > 20:
            lines.append(f"| ... | ... | ({len(lookahead_issues) - 20} more) | ... |")
    else:
        lines.append("✅ No lookahead issues detected")
    
    lines.extend([
        "",
        "---",
        "",
        "## Sample Verification Records",
        "",
        "The following records should be manually verified against BSE/NSE filings:",
        "",
        "| Symbol | Report Date | Filing Date | Available Date |",
        "|--------|-------------|-------------|----------------|",
    ])
    
    for check in sample_checks:
        lines.append(
            f"| {check['symbol']} | {check['report_date']} | "
            f"{check['filing_date']} | {check['available_date']} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])
    
    if risk["level"] == "POOR":
        lines.extend([
            "⚠️ **POOR DATA QUALITY** - High risk of lookahead bias",
            "",
            "1. Do NOT use this data for backtesting without fixes",
            "2. Manually verify filing dates for key records",
            "3. Increase reporting delay to be conservative",
            "4. Consider alternative data sources",
        ])
    elif risk["level"] == "CAUTION":
        lines.extend([
            "⚡ **CAUTION** - Moderate risk of lookahead bias",
            "",
            "1. Apply conservative reporting delay (60+ days)",
            "2. Validate sample records manually",
            "3. Be skeptical of backtest results",
        ])
    elif risk["level"] == "ACCEPTABLE":
        lines.extend([
            "📋 **ACCEPTABLE** - Minor issues found",
            "",
            "1. Current 45-day delay is appropriate",
            "2. Periodically re-audit as new data arrives",
            "3. Monitor for data quality degradation",
        ])
    else:
        lines.extend([
            "✅ **GOOD** - Data quality is high",
            "",
            "1. Continue using current configuration",
            "2. Run periodic audits",
        ])
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)
        logger.info(f"Saved PIT audit report to {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit EODHD point-in-time data quality")
    parser.add_argument(
        "--fundamentals",
        default=None,
        help="Path to EODHD fundamentals parquet",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (defaults to stdout)",
    )
    
    args = parser.parse_args()
    
    print("Loading EODHD fundamentals...", file=sys.stderr)
    df = load_eodhd_fundamentals(args.fundamentals)
    
    print(f"Loaded {len(df):,} records for {df['symbol'].nunique()} symbols", file=sys.stderr)
    
    print("Running audit...", file=sys.stderr)
    report = generate_pit_audit_report(df, args.output)
    
    if not args.output:
        print(report)
    
    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
