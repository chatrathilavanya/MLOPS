# Heart Disease API - Runbook

This guide provides instructions on how to run the full pipeline, including the API, Prometheus, and Grafana, using either **Docker Compose** (recommended for local testing) or **Kubernetes**.

---

## 1. Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** installed
- **Docker Desktop with Kubernetes Enabled** (if deploying to K8s)
- **kubectl** installed and configured to use Docker Desktop context

---

## 2. Initial Setup (Model Training)

Before deploying the API, you must train the model so that `best_pipeline.pkl` is available in the `models/` directory.

```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the training script
python src/train.py
```
This will generate the model artifacts needed by the API.

---

## 3. Option A: Run using Docker Compose (Easiest)

This spins up the API, Prometheus, and Grafana in a single command.

```bash
# Build and start all services in detached mode
docker-compose up -d --build

# Verify services are running
docker-compose ps
```

**Access Points:**
- **FastAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **Grafana:** [http://localhost:3000](http://localhost:3000) *(Username: `admin`, Password: `admin`)*

**Stop the services:**
```bash
docker-compose down
```

---

## 4. Option B: Run using Kubernetes

If you prefer to deploy the application on a Kubernetes cluster, follow these steps.

### Step 4.1: Build the Docker Image
You must first build the API image locally so Kubernetes can find it.

```bash
docker build -t heart-disease-api:latest .
```

### Step 4.2: Deploy the Core API
Apply the main API deployment and load balancer service:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Step 4.3: Deploy the Monitoring Stack
Apply the Prometheus and Grafana manifests:

```bash
# Deploy Prometheus
kubectl apply -f k8s/monitoring/prometheus-config.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/prometheus-service.yaml

# Deploy Grafana
kubectl apply -f k8s/monitoring/grafana-datasource.yaml
kubectl apply -f k8s/monitoring/grafana-dashboard.yaml
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
```

### Step 4.4: Verify Deployment
Check if all pods and services are running:

```bash
kubectl get pods
kubectl get services
```

**Access Points (via Kubernetes LoadBalancers):**
- **API Health:** `http://localhost:80/health`
- **Prometheus:** `http://localhost:9090`
- **Grafana:** `http://localhost:3000` *(Username: `admin`, Password: `admin`)*

---

## 5. Simulating Traffic

To see data populate in Grafana, you need to generate API traffic. We provide a traffic simulator that mimics diverse patient profiles.

```bash
# Ensure you are in your python virtual environment
# For Kubernetes (targeting port 80):
python scripts/traffic_simulator.py --url http://localhost:80/predict --requests 50 --delay 0.5

# For Docker Compose (targeting port 8000):
python scripts/traffic_simulator.py --url http://localhost:8000/predict --requests 50 --delay 0.5
```

## 6. Viewing the Dashboard

1. Navigate to **[http://localhost:3000](http://localhost:3000)** and log in (`admin`/`admin`).
2. Go to **Dashboards** > **Heart Disease API Monitoring**.
3. You will see real-time updates for Request Rate, Total Requests, and API Memory Usage.
