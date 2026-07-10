#!/usr/bin/env python3
"""
Inference helper: loads saved model and preprocessor, runs predictions.
Used by both the API and standalone scripts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Union

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def load_best_model():
    """Load the best trained model from disk."""
    info_path = os.path.join(MODELS_DIR, "best_model_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            "best_model_info.json not found. Run src/train.py first."
        )
    with open(info_path) as f:
        info = json.load(f)
    # Resolve model path: use stored path if it exists, else fall back to MODELS_DIR
    model_path = info["model_path"]
    if not os.path.exists(model_path):
        # Stored path may be a Windows path running in Docker/Linux — use ntpath to extract filename
        import ntpath
        model_filename = ntpath.basename(model_path)
        model_path = os.path.join(MODELS_DIR, model_filename)
    model = joblib.load(model_path)
    return model, info


def load_preprocessor():
    """Load the saved sklearn preprocessor."""
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            "preprocessor.joblib not found. Run src/train.py first."
        )
    return joblib.load(preprocessor_path)


def predict(input_data: Union[dict, pd.DataFrame]) -> dict:
    """
    Run prediction on a single patient record or a DataFrame.

    Args:
        input_data: dict with patient features or a pandas DataFrame

    Returns:
        dict with keys: prediction (int), confidence (float), model_name (str)
    """
    model, info = load_best_model()
    preprocessor = load_preprocessor()

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    X = preprocessor.transform(df)
    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][1])

    return {
        "prediction": prediction,
        "prediction_label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "confidence": round(confidence, 4),
        "model_name": info["model_name"],
    }


if __name__ == "__main__":
    # Sample patient data for quick testing
    sample = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    result = predict(sample)
    print("Prediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
