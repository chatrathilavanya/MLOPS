#!/usr/bin/env python3
"""
Evaluate.py — standalone evaluation utilities.
Generates evaluation plots and prints a full classification report.
"""

import os
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "heart.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR = os.path.join(ROOT_DIR, "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)


def evaluate_all_models():
    """Load test data and evaluate all saved models side-by-side."""
    from src.preprocess import load_and_split, load_preprocessor

    X_train_raw, X_test_raw, y_train, y_test = load_and_split(DATA_PATH)
    preprocessor = load_preprocessor()
    X_test = preprocessor.transform(X_test_raw)

    model_files = {
        "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
        "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
        "XGBoost": os.path.join(MODELS_DIR, "xgboost.joblib"),
    }

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, path) in enumerate(model_files.items()):
        if not os.path.exists(path):
            print(f"  Skipping {name} — model file not found")
            continue

        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }
        results[name] = metrics

        # ROC curve subplot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[i].plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
        axes[i].plot([0, 1], [0, 1], "k--", lw=1)
        axes[i].set_title(name, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("False Positive Rate")
        axes[i].set_ylabel("True Positive Rate")
        axes[i].legend(loc="lower right")

        print(f"\n{name}:")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    plt.suptitle("ROC Curves — All Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    roc_path = os.path.join(OUT_DIR, "06_all_models_roc.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nROC curves saved to: {roc_path}")

    # Summary table
    if results:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        header = f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
        print(header)
        print("-" * 60)
        for name, m in results.items():
            print(f"{name:<22} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
                  f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f}")

    return results


if __name__ == "__main__":
    evaluate_all_models()
