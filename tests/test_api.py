"""API endpoint tests using FastAPI TestClient (no server needed)."""

import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient

# ── Sample patient payload ────────────────────────────────────────────────────
SAMPLE_PATIENT = {
    "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
    "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
    "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0,
}

MOCK_MODEL_INFO = {
    "model_name": "Random Forest",
    "model_path": "/mock/path/random_forest.joblib",
    "metrics": {"accuracy": 0.85, "roc_auc": 0.91},
    "run_id": "mock-run-123",
}


def make_mock_model():
    """Return a mock sklearn-like model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
    return mock_model


def make_mock_preprocessor():
    """Return a mock preprocessor."""
    mock_prep = MagicMock()
    mock_prep.transform.return_value = np.zeros((1, 13))
    return mock_prep


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a test client with mocked model and preprocessor."""
    with patch("src.predict.load_best_model", return_value=(make_mock_model(), MOCK_MODEL_INFO)), \
         patch("src.predict.load_preprocessor", return_value=make_mock_preprocessor()), \
         patch("api.main.load_best_model", return_value=(make_mock_model(), MOCK_MODEL_INFO)), \
         patch("api.main.load_preprocessor", return_value=make_mock_preprocessor()):

        from api.main import app
        with TestClient(app) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """GET /health should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """Health response should have status, model_loaded, model_name keys."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data

    def test_root_returns_200(self, client):
        """GET / should return 200."""
        response = client.get("/")
        assert response.status_code == 200


class TestPredictEndpoint:

    def test_predict_returns_200(self, client):
        """POST /predict with valid payload should return 200."""
        response = client.post("/predict", json=SAMPLE_PATIENT)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        """Response should contain prediction, confidence, model_name."""
        response = client.post("/predict", json=SAMPLE_PATIENT)
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "model_name" in data
        assert "prediction_label" in data

    def test_predict_binary_output(self, client):
        """Prediction must be 0 or 1."""
        response = client.post("/predict", json=SAMPLE_PATIENT)
        data = response.json()
        assert data["prediction"] in [0, 1]

    def test_predict_confidence_range(self, client):
        """Confidence must be between 0 and 1."""
        response = client.post("/predict", json=SAMPLE_PATIENT)
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_invalid_payload_returns_422(self, client):
        """Missing required fields should return 422 Unprocessable Entity."""
        response = client.post("/predict", json={"age": 63})
        assert response.status_code == 422

    def test_predict_invalid_age_returns_422(self, client):
        """Age out of valid range (>120) should return 422."""
        bad_patient = SAMPLE_PATIENT.copy()
        bad_patient["age"] = 999
        response = client.post("/predict", json=bad_patient)
        assert response.status_code == 422

    def test_metrics_endpoint_returns_200(self, client):
        """GET /metrics should return Prometheus metrics."""
        # Trigger a request first so metrics are populated
        client.post("/predict", json=SAMPLE_PATIENT)
        response = client.get("/metrics")
        assert response.status_code == 200
