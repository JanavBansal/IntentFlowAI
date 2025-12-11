#!/usr/bin/env python3
"""
Full Production Pipeline Runner

Integrates all phases of the IntentFlow AI system:
- Phase 0: Data quality audit
- Phase 1-3: Data loading with liquidity filter
- Phase 4: 15-day semi-monthly rebalancing
- Phase 5: Macro and seasonality features
- Phase 6: Options data integration
- Phase 7: Signal reasoning with explanations
- Phase 8: Monitoring and alerting
- Phase 9: Production output generation

Usage:
    # Full pipeline (data quality + training + scoring)
    python scripts/run_full_pipeline.py --mode full
    
    # Score only (use existing model)
    python scripts/run_full_pipeline.py --mode score
    
    # Generate ranking report
    python scripts/run_full_pipeline.py --mode report
    
    # Run with monitoring
    python scripts/run_full_pipeline.py --mode full --monitor
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)


def run_data_quality_check() -> bool:
    """
    Phase 0: Run data quality checks before proceeding.
    
    Returns:
        True if data quality is acceptable, False otherwise
    """
    logger.info("=" * 60)
    logger.info("PHASE 0: Data Quality Check")
    logger.info("=" * 60)
    
    try:
        from intentflow_ai.monitoring.data_quality import run_daily_check
        
        report = run_daily_check()
        
        logger.info(f"Data Quality Level: {report.overall_level.value}")
        logger.info(f"Price Freshness: {report.price_freshness.get('status')}")
        logger.info(f"Completeness: {report.completeness.get('coverage_pct', 0):.1f}%")
        
        if not report.can_proceed:
            logger.error("Data quality issues prevent proceeding!")
            for rec in report.recommendations:
                logger.error(f"  - {rec}")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Data quality check failed: {e}")
        logger.warning("Proceeding with caution...")
        return True  # Continue but warn


def load_data_with_filters() -> Dict[str, pd.DataFrame]:
    """
    Phase 1-3: Load data with liquidity filtering.
    
    Returns:
        Dictionary with price_df, fundamentals_df, universe_df
    """
    logger.info("=" * 60)
    logger.info("PHASE 1-3: Loading Data with Filters")
    logger.info("=" * 60)
    
    # Load price data
    price_path = settings.data_dir / "raw" / "price_confirmation" / "all_prices.csv"
    if price_path.exists():
        price_df = pd.read_csv(price_path, parse_dates=["date"])
        logger.info(f"Loaded {len(price_df):,} price records for {price_df['ticker'].nunique()} tickers")
    else:
        raise FileNotFoundError(f"Price data not found: {price_path}")
    
    # Load universe/sector map
    universe_path = settings.data_dir / "static" / "sector_map.csv"
    if universe_path.exists():
        universe_df = pd.read_csv(universe_path)
        logger.info(f"Loaded universe with {len(universe_df)} tickers")
    else:
        universe_df = pd.DataFrame()
        logger.warning("Universe file not found")
    
    # Load fundamentals
    fund_path = settings.data_dir / "cache" / "fundamentals" / "eodhd_full.parquet"
    if fund_path.exists():
        fundamentals_df = pd.read_parquet(fund_path)
        logger.info(f"Loaded {len(fundamentals_df):,} fundamental records")
    else:
        fundamentals_df = pd.DataFrame()
        logger.warning("Fundamentals not found - will use available features")
    
    # Apply liquidity filter
    try:
        from intentflow_ai.data.filters.liquidity import LiquidityFilter, LiquidityConfig
        
        config = LiquidityConfig(
            min_avg_volume=100_000,
            min_avg_turnover=5_000_000,  # 50 lakh INR
            lookback_days=20,
        )
        
        liq_filter = LiquidityFilter(config)
        liquid_tickers = liq_filter.get_liquid_tickers(price_df)
        
        logger.info(f"Liquidity filter: {len(liquid_tickers)} tickers pass (of {price_df['ticker'].nunique()})")
        
        # Filter price data
        price_df = price_df[price_df["ticker"].isin(liquid_tickers)]
        
    except Exception as e:
        logger.warning(f"Liquidity filter not applied: {e}")
    
    return {
        "price_df": price_df,
        "fundamentals_df": fundamentals_df,
        "universe_df": universe_df,
    }


def build_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Phase 4-6: Build features including macro, seasonality, options.
    
    Returns:
        DataFrame with all features
    """
    logger.info("=" * 60)
    logger.info("PHASE 4-6: Building Features")
    logger.info("=" * 60)
    
    price_df = data["price_df"]
    
    # Build base features
    from intentflow_ai.features.engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    features = engineer.build(price_df)
    
    logger.info(f"Built {len(features.columns)} base features")
    
    # Add advanced features
    try:
        from intentflow_ai.features.advanced_features import build_all_advanced_features
        
        advanced = build_all_advanced_features(
            price_df,
            include_quality=True,
            include_options=True,
            include_macro=True,
            include_seasonality=True,
            include_market_cap=True,
        )
        
        if not advanced.empty:
            features = pd.concat([features, advanced], axis=1)
            logger.info(f"Added {len(advanced.columns)} advanced features")
    
    except Exception as e:
        logger.warning(f"Advanced features not added: {e}")
    
    # Add quality scores if fundamentals available
    if not data["fundamentals_df"].empty:
        try:
            from intentflow_ai.features.quality_scores import compute_quality_features
            
            quality = compute_quality_features(data["fundamentals_df"])
            logger.info(f"Computed quality scores for {len(quality)} tickers")
        except Exception as e:
            logger.warning(f"Quality scores not computed: {e}")
    
    return features


def train_model(
    features: pd.DataFrame,
    price_df: pd.DataFrame,
    use_ensemble: bool = True,
    use_wfo: bool = True,
) -> Dict[str, Any]:
    """
    Train model with optional ensemble and WFO.
    
    Returns:
        Dictionary with model and metrics
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL")
    logger.info("=" * 60)
    
    from intentflow_ai.features.labels import make_excess_label
    from intentflow_ai.pipelines.training import TrainingPipeline
    
    # Generate labels
    # Generate labels using make_excess_label
    labeled_df = make_excess_label(price_df, horizon_days=settings.signal_horizon_days, thresh=settings.target_excess_return)
    
    logger.info(f"Generated labels for {len(labeled_df)} samples")
    
    # Prepare training data
    train_df = labeled_df.copy()
    train_df = train_df.join(features, how="left")
    train_df["target"] = train_df["label"]
    train_df = train_df.dropna(subset=["target"])
    
    feature_cols = [c for c in features.columns if c in train_df.columns]
    
    if use_wfo:
        logger.info("Running Walk-Forward Optimization...")
        pipeline = TrainingPipeline()
        result = pipeline.run_wfo(
            train_df,
            feature_cols=feature_cols,
            n_splits=5,
            min_train_months=12,
        )
    else:
        # Simple train
        from intentflow_ai.modeling.trainer import LightGBMTrainer
        
        trainer = LightGBMTrainer()
        model = trainer.train(train_df[feature_cols], train_df["target"])
        
        result = {
            "model": model,
            "feature_columns": feature_cols,
            "metrics": {},
        }
    
    return result


def generate_signals(
    model_result: Dict[str, Any],
    features: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate trading signals using trained model.
    
    Returns:
        DataFrame with signals
    """
    logger.info("=" * 60)
    logger.info("GENERATING SIGNALS")
    logger.info("=" * 60)
    
    from intentflow_ai.pipelines.scoring import score_universe
    
    model = model_result.get("model")
    feature_cols = model_result.get("feature_columns", [])
    
    if model is None:
        raise ValueError("No trained model available")
    
    # Get latest data for scoring
    latest_date = price_df["date"].max()
    latest_data = price_df[price_df["date"] == latest_date].copy()
    latest_data = latest_data.join(features, how="left")
    
    # Score
    signals = score_universe(
        model=model,
        features_df=latest_data,
        feature_cols=feature_cols,
        top_k=20,
    )
    
    logger.info(f"Generated signals for {len(signals)} stocks")
    
    return signals


def generate_report(
    signals: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Phase 7: Generate ranking report with explanations.
    
    Returns:
        Report as string
    """
    logger.info("=" * 60)
    logger.info("PHASE 7: Generating Report with Reasoning")
    logger.info("=" * 60)
    
    try:
        from intentflow_ai.reasoning.signal_explainer import SignalExplainer
        
        explainer = SignalExplainer()
        
        # Generate explanations
        explanations = []
        for idx, (_, row) in enumerate(signals.head(20).iterrows()):
            ticker = row.get("ticker", "Unknown")
            proba = row.get("proba", row.get("score", 0.5))
            
            # Get features for this ticker
            features = {k: v for k, v in row.items() if isinstance(v, (int, float))}
            
            explanation = explainer.explain(
                ticker=ticker,
                features=features,
                proba=proba,
                rank=idx + 1,
                sector=sector_map.get(ticker, "Unknown") if sector_map else "Unknown",
                date=str(row.get("date", datetime.now().date())),
            )
            explanations.append(explanation)
        
        # Format report
        report_lines = [
            "=" * 70,
            "INTENTFLOW AI - SEMI-MONTHLY STOCK RANKING REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Next Rebalance: {_get_next_rebalance_date()}",
            "",
            "TOP 20 STOCK RECOMMENDATIONS",
            "-" * 70,
            "",
        ]
        
        for exp in explanations:
            report_lines.extend([
                f"#{exp.rank} {exp.ticker} ({exp.sector})",
                f"   Conviction: {exp.conviction.value}",
                f"   Signal: {exp.signal_strength:.1%}",
                f"   Suggested Allocation: {exp.suggested_allocation:.1%}",
                f"   Key Reason: {exp.key_reason}",
                "",
                f"   Technical: {exp.technical_summary}",
                f"   Fundamental: {exp.fundamental_summary}",
                f"   Sentiment: {exp.sentiment_summary}",
                "",
                f"   Key Drivers: {', '.join(exp.key_drivers[:3])}",
                f"   Risks: {', '.join(exp.risk_factors[:2])}",
                "-" * 40,
            ])
        
        report_lines.extend([
            "",
            "DISCLAIMER",
            "-" * 70,
            "This report is generated by an AI model and should not be considered",
            "financial advice. Always conduct your own research and consult with",
            "a qualified financial advisor before making investment decisions.",
            "Past performance does not guarantee future results.",
            "=" * 70,
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        # Fallback simple report
        return signals.to_string()


def _get_next_rebalance_date() -> str:
    """Calculate next semi-monthly rebalance date."""
    today = datetime.now()
    day = today.day
    
    if day < 15:
        next_date = today.replace(day=15)
    else:
        # First of next month
        if today.month == 12:
            next_date = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_date = today.replace(month=today.month + 1, day=1)
    
    return next_date.strftime("%Y-%m-%d")


def run_monitoring(signals: pd.DataFrame, model_result: Dict) -> None:
    """
    Phase 8: Run monitoring checks and generate alerts.
    """
    logger.info("=" * 60)
    logger.info("PHASE 8: Running Monitoring")
    logger.info("=" * 60)
    
    try:
        from intentflow_ai.monitoring.alerts import get_alert_manager, AlertCategory, AlertSeverity
        from intentflow_ai.monitoring.data_quality import run_daily_check
        
        manager = get_alert_manager()
        
        # Data quality check
        quality_report = run_daily_check()
        manager.check_data_quality(quality_report)
        
        # Model health check
        metrics = model_result.get("metrics", {})
        ic = metrics.get("ic", metrics.get("test_ic", 0))
        manager.check_ic(ic)
        
        # Get alert summary
        summary = manager.get_summary()
        
        if summary["total_active"] > 0:
            logger.warning(f"Active alerts: {summary['total_active']}")
            for sev, count in summary.get("by_severity", {}).items():
                logger.warning(f"  {sev}: {count}")
        else:
            logger.info("No active alerts")
            
    except Exception as e:
        logger.warning(f"Monitoring check failed: {e}")


def save_outputs(
    signals: pd.DataFrame,
    model_result: Dict,
    report: str,
    experiment_name: str = "production",
) -> None:
    """Save all outputs to experiment directory."""
    exp_dir = Path("experiments") / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save signals
    signals.to_csv(exp_dir / "top_signals.csv", index=False)
    
    # Save metrics
    metrics = model_result.get("metrics", {})
    import json
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save report
    (exp_dir / "ranking_report.txt").write_text(report)
    
    # Save model
    model = model_result.get("model")
    if model is not None:
        import joblib
        joblib.dump(model, exp_dir / "model.pkl")
    
    logger.info(f"Outputs saved to {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run IntentFlow AI production pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "score", "report"],
        default="full",
        help="Pipeline mode: full (train+score), score (use existing), report (generate report)"
    )
    parser.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="Skip data quality check"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run monitoring checks"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="experiments/production",
        help="Output directory"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        default="v_universe_sanity",
        help="Experiment to use for scoring mode"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("INTENTFLOW AI - FULL PRODUCTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Phase 0: Data quality
        if not args.skip_quality_check:
            if not run_data_quality_check():
                logger.error("Pipeline aborted due to data quality issues")
                sys.exit(1)
        
        # Phase 1-3: Load data
        data = load_data_with_filters()
        
        # Phase 4-6: Build features
        features = build_features(data)
        
        if args.mode == "full":
            # Train new model
            model_result = train_model(
                features=features,
                price_df=data["price_df"],
                use_ensemble=True,
                use_wfo=True,
            )
        else:
            # Load existing model
            exp_dir = Path("experiments") / args.experiment
            model_path = exp_dir / "lgb.pkl"
            
            if model_path.exists():
                import joblib
                model = joblib.load(model_path)
                model_result = {
                    "model": model,
                    "feature_columns": list(features.columns),
                    "metrics": {},
                }
                logger.info(f"Loaded existing model from {model_path}")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Generate signals
        signals = generate_signals(model_result, features, data["price_df"])
        
        # Phase 7: Generate report
        sector_map = None
        if not data["universe_df"].empty:
            ticker_col = "ticker_nse" if "ticker_nse" in data["universe_df"].columns else "ticker"
            sector_map = dict(zip(
                data["universe_df"][ticker_col],
                data["universe_df"]["sector"]
            ))
        
        report = generate_report(
            signals=signals,
            sector_map=sector_map,
            output_path=f"{args.output}/ranking_report.txt",
        )
        
        # Phase 8: Monitoring
        if args.monitor:
            run_monitoring(signals, model_result)
        
        # Save outputs
        save_outputs(
            signals=signals,
            model_result=model_result,
            report=report,
            experiment_name=Path(args.output).name,
        )
        
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        # Print summary
        print("\n" + "=" * 50)
        print("TOP 10 SIGNALS")
        print("=" * 50)
        print(signals.head(10).to_string())
        print()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
