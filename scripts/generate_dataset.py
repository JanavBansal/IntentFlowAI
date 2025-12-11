
import logging
from pathlib import Path
import yaml
import pandas as pd
from dataclasses import asdict

from intentflow_ai.config import Settings
from intentflow_ai.config.experiments import apply_experiment_overrides, load_experiment_config
from intentflow_ai.utils.logging import get_logger
from intentflow_ai.utils.io import load_price_parquet
from intentflow_ai.features.engineering import FeatureEngineer
from intentflow_ai.features.labels import make_excess_label

logger = get_logger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    # Load config
    exp_cfg = load_experiment_config(args.config)
    cfg = apply_experiment_overrides(Settings(), exp_cfg)
    
    # Inject into global settings AND module namespaces
    import intentflow_ai.config.settings as settings_module
    import intentflow_ai.utils.io as io_module
    import intentflow_ai.features.engineering as engineering_module
    
    settings_module.settings = cfg
    io_module.settings = cfg
    engineering_module.settings = cfg
    
    # 1. Load Prices
    logger.info("Loading prices...")
    prices = load_price_parquet(
        start_date=cfg.price_start, 
        end_date=cfg.price_end,
        cfg=cfg  # Pass explicitly where possible
    )
    
    # 2. Labeling
    logger.info(f"Generating labels (horizon={cfg.signal_horizon_days}d)...")
    labeled = make_excess_label(
        prices, 
        horizon_days=cfg.signal_horizon_days, 
        thresh=cfg.target_excess_return
    )
    
    # 3. Feature Engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer()  # Use default blocks, config from global settings
    features_df = engineer.build(labeled)
    
    # 4. Merge Labels & Features
    logger.info("merging features and labels...")
    # FeatureEngineer might return DataFrame with same index/tickers, 
    # but let's be careful. Usually FE.build() returns the dataframe with features added 
    # if it modifies in place or returns new. 
    # Looking at code, FeatureEngineer.build returns a new DF.
    # We need to ensure we have targets.
    
    # Actually, looking at engineering.py, build() preserves the index? 
    # Let's assume we need to join back labels if they were lost, 
    # BUT wait, run_training.py pipeline does:
    #   labeled_df = make_excess_label(...)
    #   train_df = self.feature_engineer.build(labeled_df)
    # So build() takes the labeled df. If build() drops columns, we might lose labels.
    # We saw in horizon analysis we had to preserve labels.
    
    # Let's check if 'label' is in features_df.
    if "label" not in features_df.columns:
        logger.info("Re-attaching labels...")
        # Prepare metadata dataframe
        metadata_cols = ["date", "ticker", "sector", "label", "excess_fwd", 
                        f"fwd_ret_{cfg.signal_horizon_days}d", "sector_fwd"]
        # Only take available columns
        available = [c for c in metadata_cols if c in labeled.columns]
        
        # Concat features with metadata
        # Since features_df was built from labeled (row-aligned order preserved)
        # we can reset index on both just to be safe, or assume alignment.
        # FeatureEngineer.build returns DF with same length.
        
        logger.info(f" restoring columns: {available}")
        features_df = pd.concat([labeled[available].reset_index(drop=True), 
                               features_df.reset_index(drop=True)], axis=1)

    # Save
    exp_dir = Path("experiments") / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = exp_dir / "train.parquet"
    features_df.to_parquet(out_path)
    logger.info(f"Saved {len(features_df)} rows to {out_path}")
    
    # Also save params.yaml for reproducibility
    params_path = exp_dir / "params.yaml"
    params_dict = {
        "settings": asdict(cfg),
        "experiment_config": exp_cfg.data if exp_cfg else {},
    }
    
    def convert_paths(obj):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, dict): return {k: convert_paths(v) for k, v in obj.items()}
        return obj

    with open(params_path, "w") as f:
        yaml.safe_dump(convert_paths(params_dict), f)

if __name__ == "__main__":
    main()
