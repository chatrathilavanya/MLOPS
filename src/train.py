#!/usr/bin/env python3
"""
Model training script with MLflow experiment tracking.
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Performs hyperparameter tuning and logs everything to MLflow.
"""

import os
import sys
import json
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

# Project paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "heart.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "mlflow_artifacts")

sys.path.insert(0, ROOT_DIR)
from src.preprocess import load_and_split, fit_and_save_preprocessor

EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR}/mlruns.db")


def compute_metrics(y_true, y_pred, y_prob):
    """Return dict of all evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = os.path.join(save_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_prob, model_name: str, save_dir: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    path = os.path.join(save_dir, f"roc_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def train_model(name: str, estimator, param_grid: dict,
                X_train, y_train, X_test, y_test,
                cv: int = 5):
    """Train a single model with grid search and log to MLflow."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=name):
        # Hyperparameter search
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        search = GridSearchCV(estimator, param_grid, cv=skf,
                              scoring="roc_auc", n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Log best params
        mlflow.log_params(search.best_params_)
        mlflow.log_param("model_name", name)
        mlflow.log_param("cv_folds", cv)

        # Cross-val score on training set
        cv_scores = cross_val_score(best_model, X_train, y_train,
                                     cv=skf, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Test evaluation
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        print(f"Test metrics: {metrics}")

        # Plots
        cm_path = plot_confusion_matrix(y_test, y_pred, name, ARTIFACTS_DIR)
        roc_path = plot_roc_curve(y_test, y_prob, name, ARTIFACTS_DIR)
        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(roc_path, "plots")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path, "models")

        # Log model to MLflow model registry
        if "XGB" in name:
            mlflow.xgboost.log_model(best_model, artifact_path="model",
                                      registered_model_name=name)
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="model",
                                      registered_model_name=name)

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        return {
            "name": name,
            "model": best_model,
            "metrics": metrics,
            "run_id": run_id,
            "model_path": model_path,
        }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading and preprocessing data...")
    X_train_raw, X_test_raw, y_train, y_test = load_and_split(DATA_PATH)
    preprocessor = fit_and_save_preprocessor(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- Model definitions ---
    models = [
        (
            "Logistic Regression",
            LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs"),
            {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"]},
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
        ),
        (
            "XGBoost",
            XGBClassifier(random_state=42, eval_metric="logloss",
                          use_label_encoder=False, verbosity=0),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
    ]

    results = []
    for name, estimator, param_grid in models:
        result = train_model(name, estimator, param_grid,
                             X_train, y_train, X_test, y_test)
        results.append(result)

    # --- Pick best model ---
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    print(f"\n{'='*50}")
    print(f"Best model: {best['name']} (ROC-AUC={best['metrics']['roc_auc']:.4f})")
    print(f"{'='*50}")

    # Save best model path for API to load
    best_model_info = {
        "model_name": best["name"],
        "model_path": best["model_path"],
        "metrics": best["metrics"],
        "run_id": best["run_id"],
    }
    info_path = os.path.join(MODELS_DIR, "best_model_info.json")
    with open(info_path, "w") as f:
        json.dump(best_model_info, f, indent=2)
    print(f"Best model info saved to {info_path}")

    print("\nTraining complete! Launch MLflow UI with:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
