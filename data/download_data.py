#!/usr/bin/env python3
"""
Download the Heart Disease UCI Dataset from the UCI ML Repository.
Saves the dataset to data/heart.csv
"""

import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "heart.csv")


def download_via_ucimlrepo() -> pd.DataFrame:
    """Download using the ucimlrepo package."""
    from ucimlrepo import fetch_ucirepo
    print("Fetching Heart Disease dataset from UCI ML Repository...")
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    # Rename target column to 'target' and binarise (0 = no disease, 1 = disease)
    target_col = y.columns[0]
    df.rename(columns={target_col: "target"}, inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    return df


def download_via_url() -> pd.DataFrame:
    """Fallback: download raw CSV from a public mirror."""
    import requests

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    col_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "target",
    ]
    print(f"Downloading from {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(response.text), header=None, names=col_names, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    return df


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}. Skipping download.")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        try:
            df = download_via_ucimlrepo()
        except Exception as e:
            print(f"ucimlrepo failed ({e}), trying direct URL download...")
            df = download_via_url()

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Dataset saved to {OUTPUT_PATH}")

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    return df


if __name__ == "__main__":
    main()
