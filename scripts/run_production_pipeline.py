#!/usr/bin/env python
"""Production-ready alpha model pipeline with full interpretability and robustness checks.

This script orchestrates the complete production workflow:
1. Feature orthogonality testing and selection
2. Regime detection and segmentation
3. Stability-optimized model training
4. Null baseline testing
5. Comprehensive stress testing
6. Signal generation with full interpretability cards
7. Drift detection and monitoring
8. Automated alerts and retrain triggers
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.config.settings import settings
from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.labels import make_excess_label
from intentflow_ai.features.orthogonality import (
    FeatureOrthogonalityAnalyzer,
    generate_orthogonality_report,
)
from intentflow_ai.modeling import LightGBMTrainer, ModelEvaluator
from intentflow_ai.modeling.explanations import SHAPExplainer, ExplanationConfig, explain_signals
from intentflow_ai.modeling.regimes import RegimeClassifier, apply_regime_filter_to_signals
from intentflow_ai.modeling.signal_cards import SignalCardGenerator
from intentflow_ai.modeling.stability import StabilityOptimizer, generate_stability_report, compare_to_baseline
from intentflow_ai.monitoring.drift_detection import DriftDetector, save_drift_report
from intentflow_ai.sanity.leakage_tests import run_null_label_test, verify_forward_alignment
from intentflow_ai.sanity.stress_tests import StressTestSuite, generate_stress_test_report
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.utils.logging import get_logger
from intentflow_ai.utils.splits import time_splits

logger = get_logger(__name__)


class ProductionPipeline:
    """Complete production-ready alpha model pipeline."""
    
    def __init__(self, experiment_name: str, output_dir: Optional[Path] = None):
        self.experiment_name = experiment_name
        self.output_dir = output_dir or Path(f"experiments/{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Initializing production pipeline",
            extra={"experiment": experiment_name, "output_dir": str(self.output_dir)}
        )
    
    def run(self):
        """Execute complete production pipeline."""
        
        logger.info("=" * 80)
        logger.info("PRODUCTION ALPHA MODEL PIPELINE - NIFTY200")
        logger.info("=" * 80)
        
        # Stage 1: Data Loading & Feature Engineering
        logger.info("Stage 1: Data Loading & Feature Engineering")
        price_panel = self._load_price_data()
        features_df, labels = self._engineer_features_and_labels(price_panel)
        
        # Save data snapshot
        universe_snapshot = price_panel[["date", "ticker"]].drop_duplicates()
        universe_snapshot.to_csv(self.output_dir / "universe_snapshot.csv", index=False)
        
        # Stage 2: Feature Orthogonality Analysis
        logger.info("Stage 2: Feature Orthogonality Analysis")
        selected_features, dropped_features = self._test_feature_orthogonality(features_df, labels)
        features_df = features_df[selected_features]
        
        # Stage 3: Create Time Splits
        logger.info("Stage 3: Creating Time-Purged Splits")
        train_mask, valid_mask, test_mask = time_splits(
            features_df.reset_index(),
            date_col="date",
            valid_start=settings.valid_start,
            test_start=settings.test_start,
            embargo_days=settings.signal_horizon_days,
            horizon_days=settings.signal_horizon_days,
        )
        
        # Stage 4: Regime Detection
        logger.info("Stage 4: Market Regime Detection")
        regime_classifier = RegimeClassifier()
        regime_data = regime_classifier.infer(price_panel)
        regime_summary = regime_classifier.get_regime_summary(regime_data)
        
        with open(self.output_dir / "regime_summary.json", "w") as f:
            json.dump(regime_summary, f, indent=2, default=float)
        
        regime_data.to_csv(self.output_dir / "regime_data.csv")
        
        # Stage 5: Stability-Optimized Training
        logger.info("Stage 5: Stability-Optimized Model Training")
        feature_array = features_df.reset_index()[selected_features]
        label_array = labels.reset_index(drop=True)
        dates = features_df.reset_index()["date"]
        
        stability_optimizer = StabilityOptimizer()
        best_config, opt_results = stability_optimizer.optimize(
            features=feature_array[train_mask | valid_mask],
            labels=label_array[train_mask | valid_mask],
            dates=dates[train_mask | valid_mask],
        )
        
        # Save stability report
        stability_report = generate_stability_report(opt_results, str(self.output_dir / "stability_report.md"))
        
        # Train final model with optimized config
        trainer = LightGBMTrainer(best_config)
        model = trainer.train(feature_array[train_mask | valid_mask], label_array[train_mask | valid_mask])
        
        # Feature importance
        feature_importance = trainer.feature_importance(model)
        pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"]).to_csv(
            self.output_dir / "feature_importance.csv", index=False
        )
        
        # Stage 6: Baseline Comparison
        logger.info("Stage 6: Baseline Model Comparison")
        baseline_comparison = compare_to_baseline(
            best_config,
            feature_array[test_mask],
            label_array[test_mask],
            baseline_type="linear"
        )
        
        with open(self.output_dir / "baseline_comparison.json", "w") as f:
            json.dump(baseline_comparison, f, indent=2, default=float)
        
        # Stage 7: Null Label Test (Leakage Detection)
        logger.info("Stage 7: Null Label Test (Leakage Detection)")
        full_frame = features_df.reset_index()
        full_frame["label"] = label_array
        
        null_test_result = run_null_label_test(
            training_frame=full_frame,
            feature_columns=selected_features,
            horizon_days=settings.signal_horizon_days,
            seed=settings.random_state,
            price_panel=price_panel,
            backtest_cfg=BacktestConfig(),
            lgbm_cfg=best_config,
            output_dir=self.output_dir / "null_test"
        )
        
        logger.info(
            "Null test results",
            extra={
                "sharpe": null_test_result.sharpe,
                "ic": null_test_result.ic,
                "rank_ic": null_test_result.rank_ic
            }
        )
        
        # Stage 8: Generate Predictions & SHAP Explanations
        logger.info("Stage 8: Generating Predictions & SHAP Explanations")
        proba, _ = trainer.predict_with_meta_label(model, feature_array)
        
        preds_df = features_df.reset_index()[["date", "ticker"]].copy()
        preds_df["proba"] = proba
        preds_df["label"] = label_array
        
        # SHAP explanations
        try:
            shap_explainer = SHAPExplainer(ExplanationConfig())
            shap_explainer.fit(model, feature_array[train_mask].head(100))
            shap_values = shap_explainer.explain(feature_array)
            shap_df = pd.DataFrame(shap_values, columns=selected_features)
        except Exception as exc:
            logger.warning(f"SHAP generation failed: {exc}")
            shap_df = pd.DataFrame()
        
        # Stage 9: Regime-Filtered Signals
        logger.info("Stage 9: Applying Regime Filters to Signals")
        test_signals = preds_df[test_mask].copy()
        test_signals = test_signals.nlargest(100, "proba")  # Top 100 signals
        
        filtered_signals = apply_regime_filter_to_signals(
            test_signals,
            regime_data,
            require_entry_allowed=True,
            min_regime_score=30.0
        )
        
        logger.info(
            "Regime filtering applied",
            extra={
                "before": len(test_signals),
                "after": len(filtered_signals),
                "filtered_pct": (1 - len(filtered_signals)/len(test_signals)) * 100
            }
        )
        
        # Stage 10: Signal Cards with Full Interpretability
        logger.info("Stage 10: Generating Signal Cards")
        card_generator = SignalCardGenerator(model_version="1.0.0")
        
        top_signals = filtered_signals.nlargest(20, "proba")
        signal_cards = card_generator.generate_cards(
            signals=top_signals,
            features=feature_array.loc[top_signals.index],
            shap_values=shap_df.loc[top_signals.index] if not shap_df.empty else None,
            regime_data=regime_data,
        )
        
        card_generator.save_cards(signal_cards, self.output_dir / "signal_cards", format="json")
        card_generator.save_cards(signal_cards, self.output_dir / "signal_cards", format="markdown")
        
        # Save top signals
        top_signals.to_csv(self.output_dir / "top_signals.csv", index=False)
        
        # Stage 11: Comprehensive Stress Testing
        logger.info("Stage 11: Comprehensive Stress Testing")
        stress_test_suite = StressTestSuite()
        stress_results = stress_test_suite.run_all_tests(
            signals=preds_df,
            prices=price_panel,
            base_config=BacktestConfig(),
            output_dir=self.output_dir / "stress_tests"
        )
        
        stress_report = generate_stress_test_report(
            stress_results,
            self.output_dir / "stress_test_report.md"
        )
        
        # Stage 12: Backtest
        logger.info("Stage 12: Running Backtest")
        bt_result = backtest_signals(preds_df, price_panel, BacktestConfig())
        
        bt_result["equity_curve"].to_csv(self.output_dir / "bt_equity.csv")
        bt_result["trades"].to_csv(self.output_dir / "bt_trades.csv", index=False)
        
        with open(self.output_dir / "bt_summary.json", "w") as f:
            json.dump(bt_result["summary"], f, indent=2, default=float)
        
        # Stage 13: Model Evaluation
        logger.info("Stage 13: Model Evaluation")
        evaluator = ModelEvaluator(horizon_days=settings.signal_horizon_days)
        
        metrics = {
            "overall": evaluator.evaluate(label_array, proba),
            "train": evaluator.evaluate(label_array[train_mask], proba[train_mask]),
            "valid": evaluator.evaluate(label_array[valid_mask], proba[valid_mask]),
            "test": evaluator.evaluate(label_array[test_mask], proba[test_mask]),
        }
        
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        
        # Stage 14: Drift Detection
        logger.info("Stage 14: Drift Detection & Monitoring")
        drift_detector = DriftDetector()
        
        alerts, drift_report = drift_detector.detect_all_drift(
            current_features=feature_array[test_mask].tail(50),
            reference_features=feature_array[train_mask].head(200),
            current_predictions=proba[test_mask].tail(50),
            reference_predictions=proba[train_mask].head(200),
            current_performance=metrics["test"],
            reference_performance=metrics["train"],
        )
        
        save_drift_report(alerts, drift_report, self.output_dir / "drift_monitoring")
        
        # Stage 15: Final Summary Report
        logger.info("Stage 15: Generating Final Summary Report")
        self._generate_summary_report(
            metrics=metrics,
            bt_summary=bt_result["summary"],
            regime_summary=regime_summary,
            stress_summary=stress_results["summary"],
            drift_summary=drift_report["summary"],
            baseline_comparison=baseline_comparison,
        )
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 80)
        
        return {
            "model": model,
            "config": best_config,
            "metrics": metrics,
            "regime_classifier": regime_classifier,
            "signal_cards": signal_cards,
        }
    
    def _load_price_data(self) -> pd.DataFrame:
        """Load price panel for universe."""
        logger.info("Loading price data")
        
        price_panel = load_price_parquet(
            allow_fallback=False,
            start_date=settings.price_start,
            end_date=settings.price_end,
            cfg=settings,
        )
        
        logger.info(
            "Price data loaded",
            extra={
                "tickers": price_panel["ticker"].nunique(),
                "dates": price_panel["date"].nunique(),
                "rows": len(price_panel)
            }
        )
        
        return price_panel
    
    def _engineer_features_and_labels(self, price_panel: pd.DataFrame):
        """Engineer features and labels."""
        logger.info("Engineering features and labels")
        
        engineer = FeatureEngineer()
        features_df = engineer.build(price_panel)
        
        # Create labels
        dataset = price_panel.join(features_df)
        labeled = make_excess_label(
            dataset,
            horizon_days=settings.signal_horizon_days,
            thresh=settings.target_excess_return,
        )
        
        labels = labeled["label"]
        
        # Verify forward alignment (leakage check)
        verify_forward_alignment(
            labeled,
            fwd_ret_col=f"fwd_ret_{settings.signal_horizon_days}d",
            horizon_days=settings.signal_horizon_days,
        )
        
        logger.info(
            "Features engineered",
            extra={
                "n_features": len(features_df.columns),
                "n_samples": len(features_df),
                "label_positive_rate": labels.mean()
            }
        )
        
        return features_df, labels
    
    def _test_feature_orthogonality(self, features: pd.DataFrame, labels: pd.Series):
        """Test and select orthogonal features."""
        logger.info("Testing feature orthogonality")
        
        analyzer = FeatureOrthogonalityAnalyzer()
        analysis = analyzer.analyze(features.reset_index()[features.columns], labels)
        
        selected, dropped = analyzer.select_orthogonal_features(
            features.reset_index()[features.columns],
            labels,
        )
        
        # Generate report
        report = generate_orthogonality_report(analysis, str(self.output_dir / "orthogonality_report.md"))
        
        logger.info(
            "Feature selection complete",
            extra={
                "original": len(features.columns),
                "selected": len(selected),
                "dropped": len(dropped)
            }
        )
        
        return selected, dropped
    
    def _generate_summary_report(
        self,
        metrics: Dict,
        bt_summary: Dict,
        regime_summary: Dict,
        stress_summary: Dict,
        drift_summary: Dict,
        baseline_comparison: Dict,
    ):
        """Generate final summary report."""
        
        lines = [
            "# Production Alpha Model - Summary Report\n",
            f"**Experiment**: {self.experiment_name}",
            f"**Date**: {pd.Timestamp.now().isoformat()}\n",
            
            "## Model Performance\n",
            f"- **Test ROC AUC**: {metrics['test'].get('roc_auc', 0):.3f}",
            f"- **Test Rank IC**: {metrics['test'].get('rank_ic', 0):.3f}",
            f"- **Test Hit Rate**: {metrics['test'].get('hit_rate', 0):.2%}\n",
            
            "## Backtest Results\n",
            f"- **CAGR**: {bt_summary.get('CAGR', 0):.1%}",
            f"- **Sharpe Ratio**: {bt_summary.get('Sharpe', 0):.2f}",
            f"- **Max Drawdown**: {bt_summary.get('maxDD', 0):.1%}",
            f"- **Win Rate**: {bt_summary.get('win_rate', 0):.1%}\n",
            
            "## Baseline Comparison\n",
            f"- **Optimized AUC**: {baseline_comparison.get('optimized_auc', 0):.3f}",
            f"- **Linear Baseline AUC**: {baseline_comparison.get('baseline_auc', 0):.3f}",
            f"- **AUC Improvement**: {baseline_comparison.get('auc_improvement', 0):+.3f}\n",
            
            "## Regime Analysis\n",
            f"- **Avg Regime Score**: {regime_summary.get('avg_regime_score', 50):.1f}/100",
            f"- **Entry Allowed**: {regime_summary.get('allow_entry_pct', 0):.1%} of days\n",
            
            "## Stress Testing\n",
            f"- **Scenarios Tested**: {stress_summary.get('total_scenarios', 0)}",
            f"- **Pass Rate**: {stress_summary.get('pass_rate', 0):.1%}",
            f"- **Worst Sharpe**: {stress_summary.get('sharpe_min', 0):.2f}\n",
            
            "## Drift Monitoring\n",
            f"- **Health Score**: {drift_summary.get('health_score', 0):.0f}/100",
            f"- **Status**: {drift_summary.get('status', 'unknown').upper()}",
            f"- **Requires Action**: {drift_summary.get('requires_action', False)}\n",
            
            "## Production Readiness\n",
            self._assess_production_readiness(metrics, bt_summary, stress_summary, drift_summary),
        ]
        
        report = "\n".join(lines)
        
        with open(self.output_dir / "PRODUCTION_SUMMARY.md", "w") as f:
            f.write(report)
        
        logger.info("Summary report generated")
    
    def _assess_production_readiness(
        self,
        metrics: Dict,
        bt_summary: Dict,
        stress_summary: Dict,
        drift_summary: Dict,
    ) -> str:
        """Assess if model is production-ready."""
        
        checks = []
        
        # Performance checks
        test_auc = metrics["test"].get("roc_auc", 0)
        checks.append(("ROC AUC > 0.55", test_auc > 0.55, test_auc))
        
        sharpe = bt_summary.get("Sharpe", 0)
        checks.append(("Sharpe > 0.5", sharpe > 0.5, sharpe))
        
        max_dd = bt_summary.get("maxDD", 0)
        checks.append(("Max DD < -25%", max_dd > -0.25, max_dd))
        
        # Stress test checks
        pass_rate = stress_summary.get("pass_rate", 0)
        checks.append(("Stress test pass rate > 50%", pass_rate > 0.5, pass_rate))
        
        # Drift checks
        health = drift_summary.get("health_score", 0)
        checks.append(("Health score > 60", health > 60, health))
        
        # Format
        lines = []
        for check, passed, value in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            lines.append(f"- {status}: {check} (value: {value})")
        
        all_passed = all(p for _, p, _ in checks)
        
        if all_passed:
            lines.append("\n**VERDICT**: ✅ **APPROVED FOR PRODUCTION**")
        else:
            lines.append("\n**VERDICT**: ❌ **NOT READY - Review failed checks**")
        
        return "\n".join(lines)


def main():
    """Run production pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production alpha model pipeline")
    parser.add_argument("--experiment", type=str, default="production_v1", help="Experiment name")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    pipeline = ProductionPipeline(args.experiment, output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()

