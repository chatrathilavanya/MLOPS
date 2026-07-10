"""
FastAPI application for Heart Disease prediction.
Endpoints:
  GET  /health   — liveness probe
  POST /predict  — run model inference
  GET  /metrics  — Prometheus metrics (auto-instrumented)
"""

import os
import sys
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from api.schemas import PatientFeatures, PredictionResponse, HealthResponse
from src.predict import predict, load_best_model, load_preprocessor

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("heart-disease-api")

# ── Model cache (loaded once at startup) ─────────────────────────────────────
_model = None
_model_info = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, release at shutdown."""
    global _model, _model_info
    logger.info("Loading model...")
    try:
        _model, _model_info = load_best_model()
        load_preprocessor()  # warm up the preprocessor cache
        logger.info(f"Model loaded: {_model_info['model_name']}")
    except FileNotFoundError as e:
        logger.warning(f"Model not loaded: {e}. Run src/train.py first.")
    yield
    logger.info("Shutting down API.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps Assignment 01 — End-to-End Heart Disease Classification API. "
        "Submit patient features and receive a prediction with confidence score."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus instrumentation ────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration}ms "
        f"client={request.client.host if request.client else 'unknown'}"
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness probe — checks if the model is loaded and ready."""
    model_loaded = _model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=_model_info["model_name"] if _model_info else "not loaded",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_endpoint(patient: PatientFeatures):
    """
    Predict heart disease risk from patient features.

    - **prediction**: 0 (no disease) or 1 (disease detected)
    - **confidence**: probability of heart disease (0.0 – 1.0)
    - **model_name**: which model produced the result
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py to train and save the model.",
        )

    try:
        input_data = patient.model_dump()
        logger.info(f"Prediction request: {input_data}")
        result = predict(input_data)
        logger.info(f"Prediction result: {result}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """API root — redirect info."""
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
