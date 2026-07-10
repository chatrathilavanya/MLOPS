"""Pydantic schemas for the FastAPI /predict endpoint."""

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Input schema — matches Heart Disease UCI dataset features."""
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: float = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: float = Field(..., ge=0, le=3, description="Thal (0=normal, 1=fixed defect, 2=reversable defect)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for /predict endpoint."""
    prediction: int = Field(..., description="Binary prediction (0=no disease, 1=disease)")
    prediction_label: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., description="Probability of heart disease (0.0 - 1.0)")
    model_name: str = Field(..., description="Name of the model used")


class HealthResponse(BaseModel):
    """Output schema for /health endpoint."""
    status: str
    model_loaded: bool
    model_name: str
