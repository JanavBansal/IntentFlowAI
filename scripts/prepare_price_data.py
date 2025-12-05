import pandas as pd
from pathlib import Path

def prepare_price_data():
    # Load prices
    print("Loading prices...")
    prices = pd.read_parquet("data/processed/prices.parquet")
    
    # Load universe with sector (using nifty200 as source)
    print("Loading sector info from nifty200...")
    sector_source = pd.read_csv("data/external/universe/nifty200.csv")
    
    # Normalize tickers
    prices['ticker'] = prices['ticker'].astype(str).str.strip().str.upper()
    sector_source['ticker'] = sector_source['ticker'].astype(str).str.strip().str.upper()
    
    # Merge sector
    print("Merging sector info...")
    # Create ticker->sector map
    if 'sector' in sector_source.columns:
        sector_map = sector_source.set_index('ticker')['sector'].to_dict()
        prices['sector'] = prices['ticker'].map(sector_map)
    else:
        print("Warning: Sector source missing 'sector' column. Defaulting to 'Unknown'.")
        prices['sector'] = 'Unknown'
        
    # Fill missing sectors
    missing_sector = prices['sector'].isna().sum()
    if missing_sector > 0:
        print(f"Warning: {missing_sector} rows missing sector. Filling with 'Unknown'.")
        prices['sector'] = prices['sector'].fillna('Unknown')

    # Save to target location
    output_path = Path("data/raw/price_confirmation/data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}...")
    prices.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    prepare_price_data()
