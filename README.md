# Heart Disease MLOps Pipeline 🫀

[![CI/CD Pipeline](https://github.com/chatrathilavanya/MLOPS/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/chatrathilavanya/MLOPS/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

**MLOps Assignment 01 — AIMLCZG523**

End-to-end MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. Features experiment tracking, CI/CD automation, Docker containerization, Kubernetes deployment, and Prometheus/Grafana monitoring.

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  UCI Dataset │───▶│  Train +     │───▶│  FastAPI        │
│  download    │    │  MLflow      │    │  /predict API   │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                 │
                   ┌──────────────┐    ┌─────────▼────────┐
                   │  Grafana     │◀───│  Prometheus       │
                   │  Dashboard   │    │  /metrics scrape  │
                   └──────────────┘    └──────────────────┘

GitHub Actions CI/CD:
  push → Lint → Tests → Train → Docker Build → Push
```

---

## 📁 Project Structure

```
mlops-assignment-01/
├── data/
│   └── download_data.py       # Dataset download script
├── src/
│   ├── preprocess.py          # sklearn preprocessing pipeline
│   ├── train.py               # Model training + MLflow tracking
│   └── predict.py             # Inference helper
├── api/
│   ├── main.py                # FastAPI application
│   └── schemas.py             # Pydantic request/response schemas
├── tests/
│   ├── test_preprocess.py     # Preprocessing unit tests
│   ├── test_model.py          # Model unit tests
│   └── test_api.py            # API endpoint tests
├── k8s/
│   ├── deployment.yaml        # Kubernetes Deployment (2 replicas)
│   └── service.yaml           # Kubernetes LoadBalancer Service
├── monitoring/
│   ├── prometheus.yml         # Prometheus scrape config
│   └── grafana/               # Grafana provisioning
├── .github/workflows/
│   └── ci-cd.yml              # GitHub Actions pipeline
├── Dockerfile                 # Multi-stage Docker image
├── docker-compose.yml         # API + Prometheus + Grafana stack
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/chatrathilavanya/MLOPS.git
cd MLOPS
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python data/download_data.py
```

### 3. Train Models

```bash
python src/train.py
```

This trains Logistic Regression, Random Forest, and XGBoost — logs everything to MLflow.

### 4. View MLflow Experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000
```

### 5. Run API Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs
```

### 6. Run Unit Tests

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## 🐳 Docker

### Build & Run

```bash
docker build -t heart-disease-api .
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models heart-disease-api
```

### Test Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

---

## 📊 Full Stack with Monitoring

```bash
# Start API + Prometheus + Grafana
docker-compose up -d

# Services:
# API:        http://localhost:8000
# API Docs:   http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```

---

## ☸️ Kubernetes (Docker Desktop)

```bash
# Enable Kubernetes in Docker Desktop Settings first

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify
kubectl get pods
kubectl get services

# Test (Docker Desktop LoadBalancer → localhost)
curl http://localhost/health
```

---

## 🤖 Models Trained

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~85% | ~0.91 |
| Random Forest | ~87% | ~0.93 |
| XGBoost | ~88% | ~0.94 |

Best model is automatically selected and used by the API.

---

## 🔄 CI/CD Pipeline (GitHub Actions)

On every push to `main`:

1. **Lint** — Ruff + Flake8 code quality checks
2. **Test** — pytest with coverage report
3. **Train** — Downloads dataset, trains all models with MLflow
4. **Build** — Docker image build, run test, push to Docker Hub

Pipeline fails loudly on any error with clear logs.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Liveness probe |
| POST | `/predict` | Heart disease prediction |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

---

## 🛠️ Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost |
| Experiment Tracking | MLflow (SQLite backend) |
| API | FastAPI + Uvicorn |
| Testing | Pytest + pytest-cov |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Orchestration | Kubernetes (Docker Desktop) |
| Monitoring | Prometheus + Grafana |

---

## 📝 License

MIT License — for academic use (AIMLCZG523 Assignment 01)
