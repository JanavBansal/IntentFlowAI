import pandas as pd
df = pd.read_parquet("data/processed/prices.parquet")
print(df.columns)
print(f"Rows: {len(df)}")
print(f"Unique Tickers: {df['ticker'].nunique()}")
print(df.head())
