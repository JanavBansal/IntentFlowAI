import pandas as pd
from intentflow_ai.features.seasonality import days_to_monthly_expiry, get_seasonality_features

def test_expiry_bug():
    # Test edge cases at year boundaries
    dates = [
        pd.Timestamp("2023-11-28"), # Before expiry
        pd.Timestamp("2024-11-29"), # Nov 29 2024 (Expiry was Nov 28). Should trigger month=13 error.
        pd.Timestamp("2023-12-29"), # Dec expiry
    ]

    print("Testing days_to_monthly_expiry...")
    for dt in dates:
        try:
            days = days_to_monthly_expiry(dt)
            print(f"{dt.date()}: {days} days to expiry")
        except Exception as e:
            print(f"ERROR for {dt.date()}: {e}")

    print("\nTesting full features...")
    for dt in dates:
        try:
            get_seasonality_features(dt)
            print(f"{dt.date()}: OK")
        except Exception as e:
            print(f"ERROR for {dt.date()}: {e}")

if __name__ == "__main__":
    test_expiry_bug()
