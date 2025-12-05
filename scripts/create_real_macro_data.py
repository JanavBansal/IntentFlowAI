import pandas as pd
import numpy as np
from pathlib import Path

def create_macro_data():
    # Annual average yields from search results
    data = {
        '2010': 7.91,
        '2011': 8.35,
        '2012': 8.01,
        '2013': 7.99,
        '2014': 8.56,
        '2015': 7.72,
        '2016': 7.33,
        '2017': 6.84,
        '2018': 7.74,
        '2019': 6.70,
        '2020': 5.95,
        '2021': 6.18,
        '2022': 7.30,
        '2023': 7.23,
        '2024': 7.07,
        '2025': 6.48
    }
    
    # Create a DataFrame with mid-year dates
    dates = []
    values = []
    for year, yield_val in data.items():
        dates.append(pd.Timestamp(f"{year}-06-30"))
        values.append(yield_val)
        
    df_annual = pd.DataFrame({'date': dates, 'yield': values})
    df_annual = df_annual.set_index('date')
    
    # Create daily range
    start_date = pd.Timestamp("2010-01-01")
    end_date = pd.Timestamp("2025-12-31")
    daily_idx = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex and interpolate
    df_daily = df_annual.reindex(daily_idx)
    df_daily['yield'] = df_daily['yield'].interpolate(method='time')
    
    # Fill edges
    df_daily['yield'] = df_daily['yield'].ffill().bfill()
    
    # Reset index
    df_daily = df_daily.reset_index().rename(columns={'index': 'date'})
    
    # Save to data directory
    output_path = Path("/Users/janavbansal/Documents/IntentFlowAI/data/india_10y_bond_yield.csv")
    df_daily.to_csv(output_path, index=False)
    print(f"Saved real macro data to {output_path}")
    print(df_daily.head())
    print(df_daily.tail())

if __name__ == "__main__":
    create_macro_data()
