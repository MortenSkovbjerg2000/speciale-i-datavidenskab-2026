import pandas as pd

# Load data
df = pd.read_csv("CRSP_top100_latest.csv")

# Convert date column
df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])

# Find earliest and latest date for each ticker
date_check = df.groupby("Ticker")["DlyCalDt"].agg(["min", "max"]).reset_index()
date_check = date_check.rename(columns={"min": "earliest_date", "max": "latest_date"})

print(date_check)

# Check if all tickers have the same earliest date
same_earliest = date_check["earliest_date"].nunique() == 1

# Check if all tickers have the same latest date
same_latest = date_check["latest_date"].nunique() == 1

print("\nSame earliest date for all tickers?", same_earliest)
print("Same latest date for all tickers?", same_latest)

if same_earliest:
    print("Common earliest date:", date_check["earliest_date"].iloc[0])

if same_latest:
    print("Common latest date:", date_check["latest_date"].iloc[0])

    