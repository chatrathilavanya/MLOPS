#!/usr/bin/env python3
"""
Standalone EDA script — generates all required visualizations.
Run: python notebooks/eda_script.py
Outputs are saved to screenshots/ directory.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "heart.csv")
OUT_DIR = os.path.join(ROOT_DIR, "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (12, 8)


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Running download_data.py...")
        from data.download_data import main
        main()
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def plot_class_distribution(df: pd.DataFrame):
    """Class balance plot."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    counts = df["target"].value_counts()
    labels = ["No Disease (0)", "Heart Disease (1)"]
    colors = ["#4CAF50", "#F44336"]

    axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 12})
    axes[0].set_title("Class Distribution (Pie)", fontsize=14, fontweight="bold")

    sns.countplot(data=df, x="target", palette=["#4CAF50", "#F44336"], ax=axes[1])
    axes[1].set_xticklabels(["No Disease", "Heart Disease"])
    axes[1].set_title("Class Distribution (Count)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Target")
    axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height())}",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_histograms(df: pd.DataFrame):
    """Histograms of all numeric features."""
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        for target, color, label in [(0, "#4CAF50", "No Disease"), (1, "#F44336", "Disease")]:
            subset = df[df["target"] == target][col].dropna()
            axes[i].hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="white")
        axes[i].set_title(col.upper(), fontsize=13, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    axes[-1].axis("off")
    plt.suptitle("Feature Distributions by Target Class", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_missing_values(df: pd.DataFrame):
    """Missing value analysis."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    if missing_df.empty:
        ax.text(0.5, 0.5, "✅ No Missing Values Found!", ha="center", va="center",
                fontsize=18, color="green", transform=ax.transAxes)
        ax.set_title("Missing Value Analysis", fontsize=14, fontweight="bold")
    else:
        missing_df["Missing Count"].plot(kind="bar", ax=ax, color="#FF7043", edgecolor="white")
        ax.set_title("Missing Values per Feature", fontsize=14, fontweight="bold")
        ax.set_ylabel("Missing Count")
        ax.set_xlabel("Feature")
        plt.xticks(rotation=45)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_missing_values.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_relationships(df: pd.DataFrame):
    """Box plots of key features by target."""
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(18, 6))

    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x="target", y=col, palette=["#4CAF50", "#F44336"],
                    ax=axes[i])
        axes[i].set_xticklabels(["No Disease", "Disease"], rotation=15)
        axes[i].set_title(col.upper(), fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")

    plt.suptitle("Feature Distribution by Target Class (Box Plots)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_feature_relationships.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_summary(df: pd.DataFrame):
    """Print dataset summary stats."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nTarget distribution:")
    print(df["target"].value_counts())
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nDescriptive statistics:")
    print(df.describe().round(2).to_string())


def main():
    print("Starting EDA...")
    df = load_data()
    print_summary(df)

    print("\nGenerating visualizations...")
    plot_class_distribution(df)
    plot_histograms(df)
    plot_correlation_heatmap(df)
    plot_missing_values(df)
    plot_feature_relationships(df)
    print(f"\nAll EDA plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
