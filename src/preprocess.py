#!/usr/bin/env python3
"""
Preprocessing pipeline for Heart Disease UCI Dataset.
Builds a reusable sklearn Pipeline + ColumnTransformer.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

# Feature groups
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names() -> list:
    """Return ordered list of feature names after transformation."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_split(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load CSV, split into train/test, return X_train, X_test, y_train, y_test."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    # Drop duplicates
    df = df.drop_duplicates()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_and_save_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on training data and save to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, PIPELINE_PATH)
    print(f"Preprocessor saved to {PIPELINE_PATH}")
    return preprocessor


def load_preprocessor() -> ColumnTransformer:
    """Load the saved preprocessor from disk."""
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found at {PIPELINE_PATH}. Run train.py first."
        )
    return joblib.load(PIPELINE_PATH)


def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocess a single inference request dict into a model-ready numpy array.
    Used by the FastAPI /predict endpoint.
    """
    preprocessor = load_preprocessor()
    df = pd.DataFrame([data])
    return preprocessor.transform(df)


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "heart.csv"
    )
    X_train, X_test, y_train, y_test = load_and_split(data_path)
    preprocessor = fit_and_save_preprocessor(X_train)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    X_train_t = preprocessor.transform(X_train)
    print(f"Transformed train shape: {X_train_t.shape}")
    print("Preprocessing pipeline built successfully!")
