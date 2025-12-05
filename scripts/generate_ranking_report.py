#!/usr/bin/env python3
"""
Generate Semi-Monthly Stock Ranking Report

Generates comprehensive stock ranking reports for stockbroker use:
- Top ranked stocks with scores
- Detailed reasoning for each pick
- Technical, fundamental, sentiment analysis
- Risk factors
- Suggested allocations
- Sector distribution

Usage:
    python scripts/generate_ranking_report.py --output reports/ranking_2024_12_15.md
    
    # Or with custom parameters
    python scripts/generate_ranking_report.py \
        --top-n 20 \
        --output reports/ranking.md \
        --format markdown
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intentflow_ai.config.settings import settings
from intentflow_ai.reasoning.signal_explainer import (
    ConvictionTier,
    SignalExplainer,
    SignalExplanation,
    generate_ranking_report as _generate_ranking_report,
)
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def load_latest_signals(
    experiment_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load latest signals from experiment directory."""
    if experiment_dir:
        signals_path = Path(experiment_dir) / "top_signals.csv"
    else:
        # Find most recent experiment
        exp_dir = settings.experiments_dir
        experiments = sorted(exp_dir.glob("*/top_signals.csv"), reverse=True)
        if not experiments:
            raise FileNotFoundError("No signals found in experiments directory")
        signals_path = experiments[0]
    
    logger.info(f"Loading signals from {signals_path}")
    df = pd.read_csv(signals_path)
    
    # Ensure required columns
    required = ["ticker", "proba"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in signals: {missing}")
    
    # Add rank if not present
    if "rank" not in df.columns:
        df = df.sort_values("proba", ascending=False)
        df["rank"] = range(1, len(df) + 1)
    
    return df


def load_features(
    experiment_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load feature data for explanation generation."""
    # Try to load from training frame
    if experiment_dir:
        feature_path = Path(experiment_dir) / "training_frame.parquet"
    else:
        exp_dir = settings.experiments_dir
        experiments = sorted(exp_dir.glob("*/training_frame.parquet"), reverse=True)
        if experiments:
            feature_path = experiments[0]
        else:
            return pd.DataFrame()
    
    if feature_path.exists():
        return pd.read_parquet(feature_path)
    return pd.DataFrame()


def load_sector_map() -> Dict[str, str]:
    """Load sector mapping."""
    sector_file = settings.data_dir / "static" / "sector_map.csv"
    if not sector_file.exists():
        return {}
    
    df = pd.read_csv(sector_file)
    if "ticker_nse" in df.columns and "sector" in df.columns:
        return dict(zip(df["ticker_nse"], df["sector"]))
    elif "ticker" in df.columns and "sector" in df.columns:
        return dict(zip(df["ticker"], df["sector"]))
    return {}


def generate_explanations(
    signals_df: pd.DataFrame,
    features_df: pd.DataFrame,
    sector_map: Dict[str, str],
    top_n: int = 20,
) -> List[SignalExplanation]:
    """Generate explanations for top signals."""
    explainer = SignalExplainer()
    explanations = []
    
    # Get top N signals
    top_signals = signals_df.nlargest(top_n, "proba")
    
    for _, row in top_signals.iterrows():
        ticker = row["ticker"]
        
        # Get features for this ticker
        features = {}
        if not features_df.empty:
            ticker_features = features_df[features_df["ticker"] == ticker]
            if not ticker_features.empty:
                features = ticker_features.iloc[-1].to_dict()
        
        # Generate explanation
        explanation = explainer.explain(
            ticker=ticker,
            features=features,
            proba=row.get("proba", 0.5),
            rank=int(row.get("rank", 999)),
            sector=row.get("sector", sector_map.get(ticker, "Unknown")),
            date=str(row.get("date", datetime.now().strftime("%Y-%m-%d"))),
            shap_values=None,  # Can be added if available
        )
        
        explanations.append(explanation)
    
    return explanations


def format_markdown_report(
    explanations: List[SignalExplanation],
    report_date: str,
    next_rebalance: str,
) -> str:
    """Format report as Markdown."""
    lines = [
        "# IntentFlow AI - Stock Ranking Report",
        "",
        f"**Generated:** {report_date}",
        f"**Next Rebalance:** {next_rebalance}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Summary stats
    high_conv = sum(1 for e in explanations if e.conviction == ConvictionTier.HIGH)
    medium_conv = sum(1 for e in explanations if e.conviction == ConvictionTier.MEDIUM)
    low_conv = sum(1 for e in explanations if e.conviction == ConvictionTier.LOW)
    
    lines.extend([
        f"- **Total Picks:** {len(explanations)}",
        f"- **High Conviction:** {high_conv}",
        f"- **Medium Conviction:** {medium_conv}",
        f"- **Low Conviction:** {low_conv}",
        "",
        "---",
        "",
        "## Top Stock Picks",
        "",
    ])
    
    # Individual picks
    for exp in explanations:
        conviction_emoji = {
            ConvictionTier.HIGH: "🟢",
            ConvictionTier.MEDIUM: "🟡",
            ConvictionTier.LOW: "🟠",
            ConvictionTier.AVOID: "🔴",
        }
        
        lines.extend([
            f"### #{exp.rank}: {exp.ticker}",
            "",
            f"**Sector:** {exp.sector} | "
            f"**Score:** {exp.probability:.2f} | "
            f"**Conviction:** {conviction_emoji.get(exp.conviction, '')} {exp.conviction.value}",
            "",
            f"**Technical:** {exp.technical_summary}",
            "",
            f"**Fundamental:** {exp.fundamental_summary}",
            "",
            f"**Sentiment:** {exp.sentiment_summary}",
            "",
            f"**Key Reason:** {exp.key_reason}",
            "",
        ])
        
        if exp.risk_factors:
            lines.append(f"**Risks:** {', '.join(exp.risk_factors)}")
            lines.append("")
        
        lines.extend([
            f"**Suggested Allocation:** {exp.suggested_allocation_pct:.1f}%",
            "",
            "---",
            "",
        ])
    
    # Sector distribution
    sectors = {}
    for exp in explanations:
        sectors[exp.sector] = sectors.get(exp.sector, 0) + 1
    
    lines.extend([
        "## Sector Distribution",
        "",
        "| Sector | Count |",
        "|--------|-------|",
    ])
    
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
        lines.append(f"| {sector} | {count} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Disclaimer",
        "",
        "*This report is generated by an AI model and is for informational purposes only. "
        "It does not constitute financial advice. Always conduct your own research and "
        "consult with a qualified financial advisor before making investment decisions.*",
        "",
    ])
    
    return "\n".join(lines)


def format_text_report(
    explanations: List[SignalExplanation],
    report_date: str,
    next_rebalance: str,
) -> str:
    """Format report as plain text."""
    return _generate_ranking_report(
        explanations=explanations,
        report_date=report_date,
        next_rebalance=next_rebalance,
        top_n=len(explanations),
    )


def format_json_report(
    explanations: List[SignalExplanation],
    report_date: str,
    next_rebalance: str,
) -> str:
    """Format report as JSON."""
    data = {
        "report_date": report_date,
        "next_rebalance": next_rebalance,
        "picks": [exp.to_dict() for exp in explanations],
        "summary": {
            "total_picks": len(explanations),
            "high_conviction": sum(1 for e in explanations if e.conviction == ConvictionTier.HIGH),
            "medium_conviction": sum(1 for e in explanations if e.conviction == ConvictionTier.MEDIUM),
            "low_conviction": sum(1 for e in explanations if e.conviction == ConvictionTier.LOW),
        },
    }
    return json.dumps(data, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Generate semi-monthly stock ranking report"
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment directory (uses latest if not specified)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top stocks to include (default: 20)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (defaults to stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "text", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--next-rebalance",
        default=None,
        help="Next rebalance date (defaults to 15 days from now)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading signals...", file=sys.stderr)
    signals_df = load_latest_signals(args.experiment)
    
    print("Loading features...", file=sys.stderr)
    features_df = load_features(args.experiment)
    
    print("Loading sector map...", file=sys.stderr)
    sector_map = load_sector_map()
    
    # Add sector to signals
    if sector_map:
        signals_df["sector"] = signals_df["ticker"].map(sector_map)
    
    # Generate explanations
    print(f"Generating explanations for top {args.top_n} signals...", file=sys.stderr)
    explanations = generate_explanations(
        signals_df=signals_df,
        features_df=features_df,
        sector_map=sector_map,
        top_n=args.top_n,
    )
    
    # Format report
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    if args.next_rebalance:
        next_rebalance = args.next_rebalance
    else:
        # Next semi-monthly date (1st or 15th)
        today = datetime.now()
        if today.day < 15:
            next_rebalance = today.replace(day=15).strftime("%Y-%m-%d")
        else:
            next_month = today.replace(day=1) + timedelta(days=32)
            next_rebalance = next_month.replace(day=1).strftime("%Y-%m-%d")
    
    if args.format == "markdown":
        report = format_markdown_report(explanations, report_date, next_rebalance)
    elif args.format == "json":
        report = format_json_report(explanations, report_date, next_rebalance)
    else:
        report = format_text_report(explanations, report_date, next_rebalance)
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)
    
    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
