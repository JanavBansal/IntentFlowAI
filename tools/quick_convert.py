#!/usr/bin/env python
"""Quick converter for NSE CSV to parquet."""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intentflow_ai.config.settings import settings

# Find all CSV files
csv_dir = ROOT / "data" / "external" / "nse_scrap" / "data"
csv_files = list(csv_dir.glob("*/*.csv"))

print(f"Found {len(csv_files)} CSV files")

all_frames = []
for i, csv_file in enumerate(csv_files):
    try:
        df = pd.read_csv(csv_file)
        
        # Strip spaces from column names
        df.columns = df.columns.str.strip()
        
        # Rename columns
        df = df.rename(columns={
            'Symbol': 'ticker',
            'Date': 'date',
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close',
            'Total Traded Quantity': 'volume',
            'Deliverable Qty': 'delivery_qty',
            '% Dly Qt to Traded Qty': 'delivery_pct',
        })
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        
        # Clean numeric columns (remove commas)
        for col in ['open', 'high', 'low', 'close', 'volume', 'delivery_qty', 'delivery_pct']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep only required columns
        cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'delivery_qty', 'delivery_pct']
        df = df[[c for c in cols if c in df.columns]]
        
        if not df.empty:
            all_frames.append(df)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(csv_files)}...")
            
    except Exception as e:
        print(f"Error in {csv_file.name}: {e}")

# Combine
print("Combining data...")
combined = pd.concat(all_frames, ignore_index=True)
combined = combined.drop_duplicates(subset=['date', 'ticker'])
combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)

# Add sector (simple mapping)
sector_map = pd.read_csv(ROOT / "data" / "static" / "sector_map.csv", on_bad_lines='skip')
if 'ticker' in sector_map.columns and 'sector' in sector_map.columns:
    sector_dict = dict(zip(sector_map['ticker'], sector_map['sector']))
    combined['sector'] = combined['ticker'].map(sector_dict).fillna('Unknown')
else:
    combined['sector'] = 'Unknown'

# Save
output = settings.data_dir / "raw" / "price_confirmation" / "data.parquet"
output.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(output, index=False)

print(f"\n✅ Success!")
print(f"Total rows: {len(combined):,}")
print(f"Unique tickers: {combined['ticker'].nunique()}")
print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
print(f"Output: {output}")
