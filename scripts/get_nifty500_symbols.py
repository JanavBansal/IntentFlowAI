import pandas as pd
from niftystocks import ns

try:
    print("Fetching NIFTY 500 symbols...")
    data = ns.get_nifty500()
    print(f"Got {len(data)} symbols")
    # Check if it's a list of strings or dicts
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], str):
            df = pd.DataFrame(data, columns=['Symbol'])
        else:
            df = pd.DataFrame(data)
            
    df.to_csv("data/external/universe/nifty500_constituents.csv", index=False)
    print("Saved to data/external/universe/nifty500_constituents.csv")
except Exception as e:
    print(f"Failed: {e}")
