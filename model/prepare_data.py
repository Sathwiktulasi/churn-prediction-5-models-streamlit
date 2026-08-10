"""
prepare_data.py
Loads the raw Telco Customer Churn dataset, cleans it, and produces a
stratified 80/20 train/test split.

Dataset: Telco Customer Churn (IBM Sample Data Set)
Source : originally published as an IBM sample dataset, widely mirrored on
         Kaggle as "Telco Customer Churn" (WA_Fn-UseC_-Telco-Customer-
         Churn.csv). 7,043 customers, 21 columns.

Run:
    python prepare_data.py
Outputs (written next to this script):
    train_data.csv    -> used only for training the 5 models
    ../test_data.csv  -> held-out 20% test split, placed at repo root
                         (this is the file that gets uploaded to the
                         Streamlit app for evaluation/demo purposes)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "Telco-Customer-Churn-full.csv")
TRAIN_OUT = os.path.join(HERE, "train_data.csv")
TEST_OUT = os.path.join(HERE, "..", "test_data.csv")

TARGET = "Churn"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop the customer identifier - it is not a predictive feature.
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as a string and has ~11 blank entries for
    # customers with tenure == 0 (brand new customers). Coerce to numeric
    # and impute with the median.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Normalize target to a clean binary integer so downstream label
    # encoding is unambiguous.
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    return df


def main():
    df = load_and_clean(RAW_PATH)

    print(f"Full dataset shape: {df.shape}")
    print(f"Feature count (excluding target): {df.shape[1] - 1}")
    print(f"Class balance:\n{df[TARGET].value_counts(normalize=True)}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET],
    )

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"Train shape: {train_df.shape} -> {TRAIN_OUT}")
    print(f"Test shape:  {test_df.shape} -> {TEST_OUT}")


if __name__ == "__main__":
    main()
