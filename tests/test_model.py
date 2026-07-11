"""Unit tests for model training and prediction logic."""

import os
import sys
import pytest
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocess import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_Xy():
    """Generate small synthetic training data."""
    import pandas as pd
    np.random.seed(0)
    n = 60
    df = pd.DataFrame({
        "age": np.random.randint(30, 80, n).astype(float),
        "sex": np.random.randint(0, 2, n).astype(float),
        "cp": np.random.randint(0, 4, n).astype(float),
        "trestbps": np.random.randint(90, 180, n).astype(float),
        "chol": np.random.randint(150, 400, n).astype(float),
        "fbs": np.random.randint(0, 2, n).astype(float),
        "restecg": np.random.randint(0, 3, n).astype(float),
        "thalach": np.random.randint(80, 200, n).astype(float),
        "exang": np.random.randint(0, 2, n).astype(float),
        "oldpeak": np.random.uniform(0, 5, n),
        "slope": np.random.randint(0, 3, n).astype(float),
        "ca": np.random.randint(0, 5, n).astype(float),
        "thal": np.random.randint(0, 4, n).astype(float),
    })
    y = np.random.randint(0, 2, n)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)
    return X, y, preprocessor


# ── Logistic Regression Tests ─────────────────────────────────────────────────

class TestLogisticRegression:

    def test_lr_trains_without_error(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        assert model.coef_ is not None

    def test_lr_predictions_binary(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1}), "Predictions must be 0 or 1"

    def test_lr_predict_proba_valid(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities must sum to 1"
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_lr_single_row_prediction(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        pred = model.predict(X[:1])
        assert pred.shape == (1,)
        assert pred[0] in [0, 1]


# ── Random Forest Tests ───────────────────────────────────────────────────────

class TestRandomForest:

    def test_rf_trains_without_error(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert model.n_estimators == 10

    def test_rf_predictions_binary(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_rf_feature_importances(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_rf_confidence_in_range(self, synthetic_Xy):
        X, y, _ = synthetic_Xy
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()
