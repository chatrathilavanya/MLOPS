# MLOps Assignment 01 - Project Report

**Name:** [Your Name]  
**Course:** Machine Learning Operations (AIMLCZG523)  
**GitHub Link:** https://github.com/chatrathilavanya/MLOPS  

## Introduction
For this assignment, I built an end-to-end machine learning pipeline to predict heart disease using the Heart Disease UCI dataset. My goal was to not just train a model, but to follow proper MLOps practices like experiment tracking, containerization, CI/CD, and deployment.

## Data Download and Exploration
I started by writing a python script (`data/download_data.py`) to fetch the Heart Disease dataset directly from the UCI repository using the `ucimlrepo` package. The dataset has 303 rows and 14 columns. The target variable is binary (0 for no disease, 1 for disease).

Once I had the data, I did some Exploratory Data Analysis (EDA) in `notebooks/01_EDA.ipynb`. I checked for missing values and found a few in the `ca` and `thal` columns, which I handled later during preprocessing. I also created some plots using seaborn and matplotlib (saved in the `screenshots/` folder):
- A count plot to see the class balance (it's pretty balanced: 164 vs 139).
- A correlation heatmap to see which features relate most to the target. For example, `thalach` (maximum heart rate) and `oldpeak` showed strong correlations with heart disease.

## Preprocessing and Modeling
Next, I set up a preprocessing pipeline using scikit-learn's `ColumnTransformer`. For the numeric columns, I used a median imputer and scaled them with `StandardScaler`. For the categorical columns, I used a most-frequent imputer followed by an `OrdinalEncoder`. I saved this entire pipeline as `preprocessor.joblib` so it can be reused exactly the same way when making predictions later.

For the modeling part, I trained three different models to see which one performed best:
1. Logistic Regression
2. Random Forest
3. XGBoost

I used MLflow to track all my experiments. I set up a local SQLite database (`mlruns.db`) as the backend for MLflow. For each model, I used `GridSearchCV` to do some basic hyperparameter tuning, and logged the accuracy, precision, recall, F1-score, and ROC-AUC metrics into MLflow. I also generated and logged confusion matrix and ROC curve plots as artifacts. 

After comparing the results, the Random Forest model turned out to be the best, achieving an ROC-AUC score of about 0.958. I saved this best model as `random_forest.joblib`.

## API Development and Testing
To serve the model, I built a REST API using FastAPI (`api/main.py`). The API has three main endpoints:
- `/health`: to check if the API is running and if the model is loaded properly.
- `/predict`: to accept patient data in JSON format and return a prediction (0 or 1) along with a confidence score.
- `/metrics`: to expose metrics for Prometheus.

I wrote 25 unit tests using `pytest` to make sure everything works. I tested the preprocessing pipeline, the model predictions, and the API endpoints. All tests are passing successfully.

## CI/CD Pipeline
To automate things, I created a GitHub Actions workflow (`.github/workflows/ci-cd.yml`). Whenever code is pushed to the repository, the pipeline runs automatically. It does four things:
1. Checks the code quality using `ruff` and `flake8`.
2. Runs all the unit tests with `pytest`.
3. Trains the models and logs them to MLflow.
4. Builds the Docker image for the API.

## Docker and Kubernetes Deployment
I containerized the FastAPI application using Docker. I wrote a `Dockerfile` that uses a lightweight python image, copies the code and models, installs the dependencies from `requirements.txt`, and runs the server. 

For deployment, I used Kubernetes (via Docker Desktop). I wrote a `deployment.yaml` to run 2 replicas of the API pods, and a `service.yaml` of type LoadBalancer to expose it on port 80. Everything deployed successfully and I was able to get predictions by sending POST requests to the Kubernetes service.

## Monitoring
Finally, I added monitoring to the API. I used `prometheus-fastapi-instrumentator` in the code to automatically collect metrics like request counts and response times. Then, I set up Prometheus and Grafana using Docker Compose (`docker-compose.yml`) to scrape these metrics and visualize them on a dashboard.

## Conclusion
This assignment helped me understand how to take a machine learning model from a simple Jupyter notebook and turn it into a production-ready application with proper tracking, testing, automation, and deployment.
