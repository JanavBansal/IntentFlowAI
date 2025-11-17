"""Comprehensive validation script for upgraded IntentFlowAI system.

This script tests all new enhancements:
1. Enhanced leakage detection
2. Aggressive feature selection
3. Ensemble training with diversity
4. Enhanced meta-labeling with risk filters
5. Black swan stress testing

Usage:
    python scripts/run_upgraded_validation.py --experiment v_universe_sanity
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from intentflow_ai.config.experiments import load_experiment_config
from intentflow_ai.config.settings import settings
from intentflow_ai.data.ingestion import prepare_training_frame
from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.selection import FeatureSelector, FeatureSelectionConfig, generate_feature_selection_report
from intentflow_ai.modeling.ensemble import DiverseEnsemble, EnsembleConfig, evaluate_ensemble_diversity, generate_ensemble_report
from intentflow_ai.meta_labeling.core import EnhancedMetaLabeler, EnhancedMetaLabelConfig
from intentflow_ai.sanity.leakage_tests import run_comprehensive_leakage_tests
from intentflow_ai.sanity.stress_tests import StressTestSuite, StressTestConfig, generate_stress_test_report
from intentflow_ai.backtest.core import BacktestConfig, backtest_signals
from intentflow_ai.utils.logging import get_logger
from intentflow_ai.utils.splits import time_splits

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run upgraded validation pipeline")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--skip-leakage", action="store_true", help="Skip leakage tests")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble training")
    parser.add_argument("--skip-stress", action="store_true", help="Skip stress testing")
    args = parser.parse_args()
    
    logger.info(f"Starting upgraded validation for experiment: {args.experiment}")
    
    # Load experiment config
    exp_cfg = load_experiment_config(args.experiment)
    output_dir = settings.experiments_dir / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ======================================================================
    # Stage 1: Data Loading & Feature Engineering
    # ======================================================================
    logger.info("Stage 1: Loading data and engineering features")
    
    training_frame = prepare_training_frame(exp_cfg)
    
    if training_frame.empty:
        logger.error("Training frame is empty!")
        return
    
    engineer = FeatureEngineer()
    features = engineer.build(training_frame)
    
    logger.info(f"Generated {len(features.columns)} features")
    
    # Extract labels and dates
    labels = training_frame["label"]
    dates = pd.to_datetime(training_frame["date"])
    
    # Split data
    train_mask, valid_mask, test_mask = time_splits(
        training_frame,
        date_col="date",
        valid_start=exp_cfg.splits["valid_start"],
        test_start=exp_cfg.splits["test_start"],
        embargo_days=10,
        horizon_days=10,
    )
    
    logger.info(
        f"Data splits - Train: {train_mask.sum()}, Valid: {valid_mask.sum()}, Test: {test_mask.sum()}"
    )
    
    # ======================================================================
    # Stage 2: Enhanced Leakage Detection
    # ======================================================================
    if not args.skip_leakage:
        logger.info("Stage 2: Running enhanced leakage detection")
        
        # Load price panel for backtesting
        price_panel = pd.read_parquet(settings.path("data/processed/price_panel.parquet"))
        
        # Backtest config
        bt_cfg = BacktestConfig(
            date_col="date",
            ticker_col="ticker",
            close_col="close",
            proba_col="proba",
            label_col="label",
            hold_days=10,
            top_k=15,
            max_weight=0.10,
            slippage_bps=10.0,
            fee_bps=1.0,
        )
        
        try:
            leak_report = run_comprehensive_leakage_tests(
                training_frame[train_mask | valid_mask],
                list(features.columns),
                horizon_days=10,
                seed=42,
                price_panel=price_panel,
                backtest_cfg=bt_cfg,
                lgbm_cfg=exp_cfg.trainer["params"],
                output_dir=output_dir / "leakage_tests",
            )
            
            # Save report
            leak_summary = leak_report.summary()
            (output_dir / "leakage_report.json").write_text(
                json.dumps(leak_summary, indent=2)
            )
            
            logger.info(f"Leakage test verdict: {leak_summary['overall_verdict']}")
            
            if not leak_report.is_clean():
                logger.warning("⚠️ Leakage detected! Review results before proceeding.")
        except Exception as exc:
            logger.warning(f"Leakage tests failed: {exc}")
    
    # ======================================================================
    # Stage 3: Aggressive Feature Selection
    # ======================================================================
    logger.info("Stage 3: Running aggressive feature selection")
    
    selector_cfg = FeatureSelectionConfig(
        max_correlation=0.80,
        max_vif=4.0,
        min_oos_ic=0.02,
        max_features=30,
        use_permutation_importance=False,  # Expensive, skip for now
        use_backward_elimination=False,  # Very expensive, skip for now
    )
    
    selector = FeatureSelector(selector_cfg)
    
    try:
        selected_features, dropped_features = selector.select_features(
            features[train_mask],
            labels[train_mask],
            dates[train_mask],
            lgbm_cfg=exp_cfg.trainer["params"],
        )
        
        logger.info(
            f"Feature selection: {len(features.columns)} → {len(selected_features)} features"
        )
        
        # Generate report
        report = generate_feature_selection_report(
            selected_features,
            dropped_features,
            output_path=str(output_dir / "feature_selection_report.md")
        )
        
        # Use selected features
        features = features[selected_features]
        
    except Exception as exc:
        logger.warning(f"Feature selection failed: {exc}, using all features")
        selected_features = list(features.columns)
    
    # ======================================================================
    # Stage 4: Ensemble Training with Diversity
    # ======================================================================
    if not args.skip_ensemble:
        logger.info("Stage 4: Training diverse ensemble")
        
        ensemble_cfg = EnsembleConfig(
            n_models=5,
            use_parameter_diversity=True,
            use_feature_diversity=True,
            use_temporal_diversity=False,  # Skip for speed
            aggregation="weighted",
            prune_low_performers=True,
            min_ic_threshold=0.01,
        )
        
        ensemble = DiverseEnsemble(ensemble_cfg)
        
        try:
            ensemble.train_ensemble(
                features[train_mask],
                labels[train_mask],
                dates[train_mask],
                base_config=exp_cfg.trainer["params"],
                validation_features=features[valid_mask],
                validation_labels=labels[valid_mask],
            )
            
            # Evaluate diversity
            diversity_metrics = evaluate_ensemble_diversity(
                ensemble,
                features[valid_mask]
            )
            
            logger.info(f"Ensemble diversity score: {diversity_metrics['diversity_score']:.3f}")
            
            # Generate report
            report = generate_ensemble_report(
                ensemble,
                diversity_metrics,
                output_path=str(output_dir / "ensemble_report.md")
            )
            
            # Get ensemble predictions
            ensemble_proba_valid, _ = ensemble.predict(features[valid_mask])
            ensemble_proba_test, _ = ensemble.predict(features[test_mask])
            
            # Save predictions
            preds_df = training_frame[["date", "ticker", "label"]].copy()
            preds_df["ensemble_proba"] = 0.0
            preds_df.loc[valid_mask, "ensemble_proba"] = ensemble_proba_valid.values
            preds_df.loc[test_mask, "ensemble_proba"] = ensemble_proba_test.values
            
            preds_df.to_csv(output_dir / "ensemble_predictions.csv", index=False)
            
        except Exception as exc:
            logger.error(f"Ensemble training failed: {exc}")
            return
    else:
        logger.info("Skipping ensemble training (use --skip-ensemble to skip)")
        # Use simple model as fallback
        from intentflow_ai.modeling.trainer import LightGBMTrainer
        trainer = LightGBMTrainer(exp_cfg.trainer["params"])
        model = trainer.train(features[train_mask], labels[train_mask])
        
        ensemble_proba_valid, _ = trainer.predict_with_meta_label(model, features[valid_mask])
        ensemble_proba_test, _ = trainer.predict_with_meta_label(model, features[test_mask])
    
    # ======================================================================
    # Stage 5: Enhanced Meta-Labeling & Risk Filtering
    # ======================================================================
    logger.info("Stage 5: Applying enhanced meta-labeling and risk filters")
    
    meta_cfg = EnhancedMetaLabelConfig(
        enabled=True,
        max_stock_drawdown=-0.15,
        max_portfolio_drawdown=-0.10,
        target_win_rate=0.55,
        min_risk_reward=1.5,
        use_kelly_sizing=False,  # Conservative
        block_high_vol_regime=True,
        min_pattern_success_rate=0.50,
    )
    
    meta_labeler = EnhancedMetaLabeler(meta_cfg)
    
    try:
        # Train meta-model
        meta_result = meta_labeler.train(
            training_frame[train_mask],
            pd.Series(ensemble_proba_valid.values if isinstance(ensemble_proba_valid, pd.Series) else ensemble_proba_valid, index=training_frame[train_mask].index)
        )
        
        # Predict on validation set
        val_frame = training_frame[valid_mask]
        val_base_proba = pd.Series(ensemble_proba_valid.values if isinstance(ensemble_proba_valid, pd.Series) else ensemble_proba_valid, index=val_frame.index)
        
        meta_proba_valid = meta_labeler.predict(
            meta_result["model"],
            val_frame,
            val_base_proba,
            meta_result["feature_columns"]
        )
        
        # Apply risk filters
        allowed = meta_labeler.apply_risk_filters(
            val_frame,
            val_base_proba,
            meta_proba_valid
        )
        
        logger.info(
            f"Risk filters: {allowed.sum()}/{len(allowed)} signals allowed "
            f"({allowed.sum()/len(allowed)*100:.1f}%)"
        )
        
    except Exception as exc:
        logger.warning(f"Meta-labeling failed: {exc}")
        allowed = pd.Series(True, index=training_frame[valid_mask].index)
    
    # ======================================================================
    # Stage 6: Black Swan Stress Testing
    # ======================================================================
    if not args.skip_stress:
        logger.info("Stage 6: Running black swan stress tests")
        
        # Prepare signals for backtest
        signals = training_frame[test_mask][["date", "ticker", "label"]].copy()
        signals["proba"] = ensemble_proba_test.values if not isinstance(ensemble_proba_test, pd.Series) else ensemble_proba_test.values
        
        # Load price panel
        price_panel = pd.read_parquet(settings.path("data/processed/price_panel.parquet"))
        
        # Backtest config
        bt_cfg = BacktestConfig(
            date_col="date",
            ticker_col="ticker",
            close_col="close",
            proba_col="proba",
            label_col="label",
            hold_days=10,
            top_k=15,
            max_weight=0.10,
            slippage_bps=10.0,
            fee_bps=1.0,
        )
        
        # Stress test config
        stress_cfg = StressTestConfig(
            test_black_swans=True,
            run_monte_carlo=False,  # Skip for speed
            test_parameter_sensitivity=True,
        )
        
        try:
            suite = StressTestSuite(stress_cfg)
            stress_results = suite.run_all_tests(
                signals,
                price_panel,
                bt_cfg,
                output_dir=output_dir / "stress_tests"
            )
            
            # Generate report
            report = generate_stress_test_report(
                stress_results,
                output_path=str(output_dir / "stress_test_report.md")
            )
            
            pass_rate = stress_results["summary"]["pass_rate"]
            logger.info(f"Stress test pass rate: {pass_rate:.1%}")
            
            if pass_rate < 0.70:
                logger.warning("⚠️ Low stress test pass rate! System may not be robust.")
            else:
                logger.info("✅ System passed stress testing")
                
        except Exception as exc:
            logger.error(f"Stress testing failed: {exc}")
    
    # ======================================================================
    # Stage 7: Generate Summary Report
    # ======================================================================
    logger.info("Stage 7: Generating summary report")
    
    # Compute final metrics on test set
    from intentflow_ai.modeling.evaluation import ModelEvaluator
    
    evaluator = ModelEvaluator(horizon_days=10)
    test_metrics = evaluator.evaluate(
        labels[test_mask],
        ensemble_proba_test,
        dates=dates[test_mask]
    )
    
    summary = {
        "experiment": args.experiment,
        "timestamp": pd.Timestamp.now().isoformat(),
        "feature_count_original": len(engineer.build(training_frame).columns),
        "feature_count_selected": len(selected_features),
        "feature_reduction_pct": (1 - len(selected_features) / len(engineer.build(training_frame).columns)) * 100,
        "test_metrics": {
            "roc_auc": float(test_metrics.get("roc_auc", 0.0)),
            "ic": float(test_metrics.get("ic", 0.0)),
            "rank_ic": float(test_metrics.get("rank_ic", 0.0)),
        },
        "ensemble": {
            "n_models": len(ensemble.models) if not args.skip_ensemble else 1,
            "diversity_score": diversity_metrics.get("diversity_score", 0.0) if not args.skip_ensemble else 0.0,
        },
        "risk_filtering": {
            "signals_total": len(allowed),
            "signals_allowed": int(allowed.sum()),
            "block_rate_pct": (1 - allowed.sum() / len(allowed)) * 100,
        },
    }
    
    # Add stress test summary if available
    if not args.skip_stress and 'stress_results' in locals():
        summary["stress_tests"] = {
            "total_scenarios": stress_results["summary"]["total_scenarios"],
            "passed": stress_results["summary"]["passed"],
            "pass_rate": stress_results["summary"]["pass_rate"],
        }
    
    # Save summary
    (output_dir / "upgrade_validation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    
    # Print summary
    print("\n" + "="*80)
    print("UPGRADE VALIDATION SUMMARY")
    print("="*80)
    print(f"Experiment: {args.experiment}")
    print(f"Features: {summary['feature_count_original']} → {summary['feature_count_selected']} "
          f"({summary['feature_reduction_pct']:.1f}% reduction)")
    print(f"\nTest Metrics:")
    print(f"  ROC AUC: {summary['test_metrics']['roc_auc']:.4f}")
    print(f"  IC: {summary['test_metrics']['ic']:.4f}")
    print(f"  Rank IC: {summary['test_metrics']['rank_ic']:.4f}")
    print(f"\nEnsemble:")
    print(f"  Models: {summary['ensemble']['n_models']}")
    print(f"  Diversity: {summary['ensemble']['diversity_score']:.3f}")
    print(f"\nRisk Filtering:")
    print(f"  Allowed: {summary['risk_filtering']['signals_allowed']}/{summary['risk_filtering']['signals_total']} "
          f"({100-summary['risk_filtering']['block_rate_pct']:.1f}%)")
    
    if 'stress_tests' in summary:
        print(f"\nStress Tests:")
        print(f"  Pass Rate: {summary['stress_tests']['pass_rate']:.1%} "
              f"({summary['stress_tests']['passed']}/{summary['stress_tests']['total_scenarios']})")
    
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("="*80 + "\n")
    
    logger.info("Upgrade validation complete!")


if __name__ == "__main__":
    main()

