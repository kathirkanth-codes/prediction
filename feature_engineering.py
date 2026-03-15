import pandas as pd
import numpy as np


def create_features(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    # -----------------------------------
    # keep only spending transactions
    # -----------------------------------
    df = df[df["amount"] < 0]

    # -----------------------------------
    # convert negative amount to positive spending
    # -----------------------------------
    df["spending"] = df["amount"].abs()

    # -----------------------------------
    # monthly category spending
    # -----------------------------------
    monthly = (
        df.groupby(["month", "category"])["spending"]
        .sum()
        .reset_index()
    )

    # sort dataset
    monthly = monthly.sort_values(["category", "month"])

    # -----------------------------------
    # Lag features (previous months spending)
    # -----------------------------------
    monthly["lag1"] = monthly.groupby("category")["spending"].shift(1)
    monthly["lag2"] = monthly.groupby("category")["spending"].shift(2)
    monthly["lag3"] = monthly.groupby("category")["spending"].shift(3)

    # -----------------------------------
    # Rolling average (last 3 months)
    # -----------------------------------
    monthly["rolling_avg"] = (
        monthly.groupby("category")["spending"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------------
    # Rolling standard deviation (spending stability)
    # -----------------------------------
    monthly["rolling_std"] = (
        monthly.groupby("category")["spending"]
        .rolling(window=3, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------------
    # Trend (increase or decrease from last month)
    # -----------------------------------
    monthly["trend"] = monthly["spending"] - monthly["lag1"]

    # -----------------------------------
    # Momentum (difference from rolling average)
    # -----------------------------------
    monthly["momentum"] = monthly["spending"] - monthly["rolling_avg"]

    # -----------------------------------
    # Seasonality features (month encoding)
    # -----------------------------------
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)

    # -----------------------------------
    # Fill missing values
    # -----------------------------------
    monthly = monthly.fillna(0)

    # -----------------------------------
    # Save dataset
    # -----------------------------------
    monthly.to_csv(output_csv, index=False)

    print("Feature engineering completed.")
    print("Rows created:", len(monthly))

    return monthly