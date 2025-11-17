#!/usr/bin/env python
"""Fetch NSE bhavcopy data from GitHub repository: https://github.com/tilak999/NSE-Data-bank

This script clones or updates the NSE-Data-bank repository and processes the bhavcopy data
for use in IntentFlowAI. The repository is updated daily with latest NSE equity bhavcopy data.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings
from intentflow_ai.utils.logging import get_logger

logger = get_logger(__name__)

NSE_DATA_BANK_REPO = "https://github.com/tilak999/NSE-Data-bank.git"
DEFAULT_CLONE_DIR = ROOT / "data" / "external" / "nse_data_bank"


def clone_or_update_repo(clone_dir: Path, force_clone: bool = False) -> bool:
    """Clone or update the NSE-Data-bank repository.
    
    Args:
        clone_dir: Directory where to clone the repository
        force_clone: If True, remove existing directory and clone fresh
    
    Returns:
        True if successful, False otherwise
    """
    clone_dir = Path(clone_dir)
    
    if force_clone and clone_dir.exists():
        logger.info(f"Removing existing repository at {clone_dir}")
        shutil.rmtree(clone_dir)
    
    if clone_dir.exists() and (clone_dir / ".git").exists():
        # Repository exists, try to update
        logger.info(f"Updating existing repository at {clone_dir}")
        try:
            subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=clone_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Repository updated successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git pull failed: {e.stderr}. Attempting fresh clone...")
            shutil.rmtree(clone_dir)
    
    # Clone the repository
    logger.info(f"Cloning NSE-Data-bank repository to {clone_dir}")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["git", "clone", NSE_DATA_BANK_REPO, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Repository cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e.stderr}")
        return False


def find_bhavcopy_files(repo_dir: Path, data_dir: str = "data") -> list[Path]:
    """Find all bhavcopy CSV files in the repository.
    
    Args:
        repo_dir: Root directory of the cloned repository
        data_dir: Subdirectory containing data (default: "data")
    
    Returns:
        List of paths to bhavcopy CSV files
    """
    data_path = repo_dir / data_dir
    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_path}")
        return []
    
    # Find all CSV files
    csv_files = sorted(data_path.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {data_path}")
    return csv_files


def process_bhavcopy_file(csv_path: Path) -> Optional[pd.DataFrame]:
    """Process a single bhavcopy CSV file into standardized format.
    
    Args:
        csv_path: Path to bhavcopy CSV file
    
    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume
        Returns None if file cannot be processed
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        logger.debug(f"Failed to read {csv_path.name}: {e}")
        return None
    
    # Normalize column names (case-insensitive)
    col_map = {col.upper(): col for col in df.columns}
    
    # Map NSE bhavcopy column names to standard names
    # The actual files use: SYMBOL, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY, DELIV_QTY
    rename_map = {}
    
    # Ticker mapping
    if "SYMBOL" in col_map:
        rename_map[col_map["SYMBOL"]] = "ticker"
    elif "TRADINGSYMBOL" in col_map:
        rename_map[col_map["TRADINGSYMBOL"]] = "ticker"
    
    # Date mapping
    if "DATE1" in col_map:
        rename_map[col_map["DATE1"]] = "date"
    elif "TIMESTAMP" in col_map:
        rename_map[col_map["TIMESTAMP"]] = "date"
    elif "DATE" in col_map:
        rename_map[col_map["DATE"]] = "date"
    
    # Price mappings
    if "OPEN_PRICE" in col_map:
        rename_map[col_map["OPEN_PRICE"]] = "open"
    elif "OPEN" in col_map:
        rename_map[col_map["OPEN"]] = "open"
    
    if "HIGH_PRICE" in col_map:
        rename_map[col_map["HIGH_PRICE"]] = "high"
    elif "HIGH" in col_map:
        rename_map[col_map["HIGH"]] = "high"
    
    if "LOW_PRICE" in col_map:
        rename_map[col_map["LOW_PRICE"]] = "low"
    elif "LOW" in col_map:
        rename_map[col_map["LOW"]] = "low"
    
    if "CLOSE_PRICE" in col_map:
        rename_map[col_map["CLOSE_PRICE"]] = "close"
    elif "CLOSE" in col_map:
        rename_map[col_map["CLOSE"]] = "close"
    
    # Volume mapping
    if "TTL_TRD_QNTY" in col_map:
        rename_map[col_map["TTL_TRD_QNTY"]] = "volume"
    elif "TOTTRDQTY" in col_map:
        rename_map[col_map["TOTTRDQTY"]] = "volume"
    elif "VOLUME" in col_map:
        rename_map[col_map["VOLUME"]] = "volume"
    
    # Delivery quantity (optional)
    if "DELIV_QTY" in col_map:
        rename_map[col_map["DELIV_QTY"]] = "delivery_qty"
    elif "DELIVERYQTY" in col_map:
        rename_map[col_map["DELIVERYQTY"]] = "delivery_qty"
    
    if not rename_map:
        logger.debug(f"No recognized columns in {csv_path.name}")
        return None
    
    df = df.rename(columns=rename_map)
    
    # Check required columns
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.debug(f"{csv_path.name} missing columns: {missing}")
        return None
    
    # Extract and clean data
    result = df[list(required)].copy()
    
    # Convert date
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"])
    
    # Clean ticker
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    
    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    
    # Filter out invalid rows
    result = result.dropna(subset=["open", "high", "low", "close", "volume"])
    result = result[
        (result["open"] > 0)
        & (result["high"] > 0)
        & (result["low"] > 0)
        & (result["close"] > 0)
        & (result["volume"] >= 0)
    ]
    
    if result.empty:
        return None
    
    return result


def add_sector_info(df: pd.DataFrame, sector_map_path: Optional[Path] = None) -> pd.DataFrame:
    """Add sector information to the dataframe.
    
    Args:
        df: DataFrame with ticker column
        sector_map_path: Optional path to sector mapping CSV
    
    Returns:
        DataFrame with sector column added
    """
    if sector_map_path is None:
        sector_map_path = ROOT / "data" / "static" / "sector_map.csv"
    
    # Try to load sector map
    if sector_map_path.exists():
        try:
            sectors = pd.read_csv(sector_map_path)
            if "ticker" in sectors.columns and "sector" in sectors.columns:
                sectors = sectors[["ticker", "sector"]].copy()
                sectors["ticker"] = sectors["ticker"].astype(str).str.strip().str.upper()
                df = df.merge(sectors, on="ticker", how="left")
                logger.info(f"Added sectors from {sector_map_path}: {df['sector'].notna().sum()}/{len(df)} rows have sectors")
            else:
                logger.warning(f"Sector map missing required columns. Expected: ticker, sector. Found: {sectors.columns.tolist()}")
        except Exception as e:
            logger.warning(f"Failed to load sector map: {e}")
    
    # If still missing sectors, try universe file
    if "sector" not in df.columns or df["sector"].isna().all():
        universe_path = ROOT / "data" / "external" / "universe" / "nifty200.csv"
        if universe_path.exists():
            try:
                universe = pd.read_csv(universe_path)
                # Try different column name variations
                ticker_col = None
                sector_col = None
                for col in universe.columns:
                    col_lower = col.lower()
                    if ticker_col is None and ("ticker" in col_lower or "symbol" in col_lower):
                        ticker_col = col
                    if sector_col is None and "sector" in col_lower:
                        sector_col = col
                
                if ticker_col and sector_col:
                    sectors = universe[[ticker_col, sector_col]].rename(columns={ticker_col: "ticker", sector_col: "sector"})
                    sectors["ticker"] = sectors["ticker"].astype(str).str.strip().str.upper()
                    df = df.merge(sectors, on="ticker", how="left")
                    logger.info(f"Added sectors from universe file: {df['sector'].notna().sum()}/{len(df)} rows have sectors")
            except Exception as e:
                logger.warning(f"Failed to load universe file: {e}")
    
    # If still no sectors, assign "Unknown"
    if "sector" not in df.columns or df["sector"].isna().all():
        df["sector"] = "Unknown"
        logger.warning("No sector information found. Assigning 'Unknown' to all tickers.")
    else:
        # Fill missing sectors with "Unknown"
        df["sector"] = df["sector"].fillna("Unknown")
    
    return df


def process_all_bhavcopy(
    repo_dir: Path,
    output_path: Optional[Path] = None,
    max_files: Optional[int] = None,
    sector_map_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Process all bhavcopy files from the repository.
    
    Args:
        repo_dir: Root directory of the cloned repository
        output_path: Optional path to save processed parquet file
        max_files: Optional limit on number of files to process (for testing)
    
    Returns:
        Combined DataFrame with all processed bhavcopy data
    """
    csv_files = find_bhavcopy_files(repo_dir)
    
    if not csv_files:
        logger.warning("No bhavcopy files found")
        return pd.DataFrame()
    
    if max_files:
        csv_files = csv_files[:max_files]
        logger.info(f"Processing first {max_files} files for testing")
    
    frames = []
    processed = 0
    failed = 0
    
    for csv_file in csv_files:
        df = process_bhavcopy_file(csv_file)
        if df is not None and not df.empty:
            frames.append(df)
            processed += 1
        else:
            failed += 1
        
        if processed % 100 == 0:
            logger.info(f"Processed {processed} files, {failed} failed")
    
    if not frames:
        logger.warning("No valid bhavcopy data processed")
        return pd.DataFrame()
    
    # Combine all frames
    combined = pd.concat(frames, ignore_index=True)
    
    # Remove duplicates and sort
    combined = combined.drop_duplicates(subset=["date", "ticker"])
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Add sector information
    combined = add_sector_info(combined, sector_map_path)
    
    logger.info(
        f"Processed {processed} files successfully, {failed} failed",
        extra={
            "total_rows": len(combined),
            "tickers": combined["ticker"].nunique(),
            "date_range": f"{combined['date'].min().date()} to {combined['date'].max().date()}",
        },
    )
    
    # Save to parquet if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and process NSE bhavcopy data from GitHub repository"
    )
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=DEFAULT_CLONE_DIR,
        help=f"Directory to clone repository (default: {DEFAULT_CLONE_DIR})",
    )
    parser.add_argument(
        "--force-clone",
        action="store_true",
        help="Force fresh clone (removes existing directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for processed parquet file (default: data/raw/price_confirmation/data.parquet)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning/updating, use existing repository",
    )
    parser.add_argument(
        "--sector-map",
        type=Path,
        help="Path to sector mapping CSV file (default: data/static/sector_map.csv)",
    )
    
    args = parser.parse_args()
    
    # Clone or update repository
    if not args.skip_clone:
        success = clone_or_update_repo(args.clone_dir, force_clone=args.force_clone)
        if not success:
            logger.error("Failed to clone/update repository")
            sys.exit(1)
    elif not args.clone_dir.exists():
        logger.error(f"Repository not found at {args.clone_dir}. Remove --skip-clone to clone it.")
        sys.exit(1)
    
    # Process bhavcopy files
    output_path = args.output or (settings.data_dir / "raw" / "price_confirmation" / "data.parquet")
    result = process_all_bhavcopy(
        args.clone_dir,
        output_path=output_path,
        max_files=args.max_files,
        sector_map_path=args.sector_map,
    )
    
    if result.empty:
        logger.error("No data processed. Check repository structure and file formats.")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("NSE BHAVCOPY DATA PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total rows: {len(result):,}")
    print(f"Unique tickers: {result['ticker'].nunique()}")
    print(f"Date range: {result['date'].min().date()} to {result['date'].max().date()}")
    print(f"Output saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

