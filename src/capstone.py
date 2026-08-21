from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


UCI_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

TARGET = "quality"
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]


def download_wine_quality(destination: Path, timeout: int = 30) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        UCI_WINE_URL,
        timeout=timeout,
        headers={"User-Agent": "week6-integrative-capstone/1.0"},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_wine_csv(path: Path) -> pd.DataFrame:
    """Load either the official semicolon-delimited UCI file or the comma-delimited smoke fixture."""
    path = Path(path)
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


def clean_wine_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    before = len(work)

    # Normalize column names.
    work.columns = (
        work.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    work = work.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(work)

    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    missing_before = int(work.isna().sum().sum())
    work = work.dropna().reset_index(drop=True)
    missing_rows_removed = before - duplicates_removed - len(work)

    # Plausibility checks for quality-control ranges.
    valid = (
        work["fixed_acidity"].gt(0)
        & work["volatile_acidity"].gt(0)
        & work["citric_acid"].between(0, 1)
        & work["residual_sugar"].gt(0)
        & work["chlorides"].gt(0)
        & work["free_sulfur_dioxide"].ge(0)
        & work["total_sulfur_dioxide"].ge(0)
        & work["density"].between(0.98, 1.02)
        & work["ph"].between(2.0, 5.0)
        & work["sulphates"].gt(0)
        & work["alcohol"].gt(0)
        & work["quality"].between(0, 10)
    )
    invalid_range_rows = int((~valid).sum())
    work = work.loc[valid].reset_index(drop=True)

    if work.empty:
        raise ValueError("Cleaning removed all observations.")

    report = {
        "rows_before": int(before),
        "rows_after": int(len(work)),
        "duplicate_rows_removed": int(duplicates_removed),
        "missing_cells_before_drop": int(missing_before),
        "rows_removed_for_missing_values": int(missing_rows_removed),
        "rows_removed_for_range_validation": int(invalid_range_rows),
        "missing_cells_after": int(work.isna().sum().sum()),
        "duplicate_rows_after": int(work.duplicated().sum()),
        "quality_min": float(work[TARGET].min()),
        "quality_max": float(work[TARGET].max()),
    }

    assert report["missing_cells_after"] == 0
    assert report["duplicate_rows_after"] == 0
    assert work[TARGET].between(0, 10).all()

    return work, report


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = df[TARGET].astype(float).copy()

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Modeling data contains missing values.")
    return X, y


def eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.describe().T.reset_index().rename(columns={"index": "feature"})
    return summary


def run_eda_figures(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1 quality distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, alpha=0.85)
    ax.set_title("Wine Quality Distribution")
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir/"01_quality_distribution.png", dpi=160)
    plt.close(fig)

    # 2 feature distributions
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.ravel()
    for ax, feature in zip(axes, FEATURES):
        ax.hist(df[feature], bins=25, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.set_title(feature.replace("_", " ").title())
    axes[-1].axis("off")
    fig.suptitle("Physicochemical Feature Distributions")
    fig.tight_layout()
    fig.savefig(output_dir/"02_feature_distributions.png", dpi=160)
    plt.close(fig)

    # 3 correlation
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir/"03_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    # 4 quality vs alcohol
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="alcohol", ax=ax)
    ax.set_title("Alcohol Distribution by Wine Quality")
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Alcohol (%)")
    fig.tight_layout()
    fig.savefig(output_dir/"04_quality_vs_alcohol.png", dpi=160)
    plt.close(fig)


def build_models() -> dict:
    return {
        "Dummy Mean": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
            n_estimators=200,
            learning_rate=0.04,
            max_depth=2,
            loss="squared_error",
        ),
    }


def regression_cv(models: dict, X: pd.DataFrame, y: pd.Series, cv) -> pd.DataFrame:
    rows = []
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": name,
                "mae_mean": -float(np.mean(scores["test_mae"])),
                "mae_std": float(np.std(scores["test_mae"], ddof=1)),
                "rmse_mean": -float(np.mean(scores["test_rmse"])),
                "rmse_std": float(np.std(scores["test_rmse"], ddof=1)),
                "r2_mean": float(np.mean(scores["test_r2"])),
                "r2_std": float(np.std(scores["test_r2"], ddof=1)),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


def fit_and_evaluate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse,
                "r2": float(r2_score(y_test, pred)),
            }
        )
        fitted[name] = model

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True), fitted


def feature_importance_table(model, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=12,
        random_state=42,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_model_comparison(results: pd.DataFrame, output_path: Path) -> None:
    ordered = results.sort_values("rmse").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(ordered))
    ax.barh(y, ordered["rmse"].to_numpy(), alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"].tolist())
    ax.invert_yaxis()
    ax.set_title("Held-Out Test RMSE by Regression Model")
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    top = importance.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_title("Permutation Feature Importance — Best Regressor")
    ax.set_xlabel("Mean change in negative RMSE")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def kmeans_analysis(
    X: pd.DataFrame,
    output_dir: Path,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int, KMeans, StandardScaler, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metric_rows = []
    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=random_state,
        )
        labels = model.fit_predict(X_scaled)
        metric_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X_scaled, labels)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    best_k = int(metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])

    final_model = KMeans(
        n_clusters=best_k,
        n_init=20,
        random_state=random_state,
    )
    labels = final_model.fit_predict(X_scaled)

    profile = pd.DataFrame(X_scaled, columns=X.columns)
    profile["cluster"] = labels
    profile = profile.groupby("cluster").mean()

    sizes = (
        pd.Series(labels, name="cluster")
        .value_counts()
        .sort_index()
        .rename("count")
        .reset_index()
    )
    sizes["percentage"] = sizes["count"] / sizes["count"].sum() * 100

    # silhouette plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o")
    ax.set_title("K-Means Silhouette Score")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    fig.tight_layout()
    fig.savefig(output_dir/"07_kmeans_silhouette.png", dpi=160)
    plt.close(fig)

    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    projection = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        s=35,
        alpha=0.85,
    )
    ax.set_title(
        f"K-Means Clusters in PCA Space "
        f"({pca.explained_variance_ratio_[0]*100:.1f}% + "
        f"{pca.explained_variance_ratio_[1]*100:.1f}% variance)"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_dir/"08_kmeans_pca.png", dpi=160)
    plt.close(fig)

    # profile heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(profile, cmap="vlag", center=0, ax=ax)
    ax.set_title("Cluster Profile — Mean Standardized Features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(output_dir/"09_cluster_profile_heatmap.png", dpi=160)
    plt.close(fig)

    return metrics, best_k, final_model, scaler, profile, sizes


def save_json(data: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_capstone_summary(
    path: Path,
    metadata: dict,
    cv_results: pd.DataFrame,
    test_results: pd.DataFrame,
    best_model_name: str,
    best_k: int,
    feature_importance: pd.DataFrame,
) -> None:
    best_test = test_results.sort_values("rmse").iloc[0]
    top_features = feature_importance.head(5)["feature"].tolist()
    summary = f"""# Week 6 Capstone Summary

## Project
**Integrative Wine Quality Data Science Capstone**

## Problem statement
Use physicochemical measurements of red wine to predict quality and identify unsupervised wine segments.

## Data pipeline
- Public source: UCI Wine Quality dataset.
- Cleaning: numeric conversion, duplicate removal, missing-value handling, range validation.
- EDA: distributions, correlation analysis, and quality-by-alcohol comparison.
- Supervised learning: Dummy baseline, Ridge, Random Forest, Gradient Boosting.
- Unsupervised learning: K-Means with standardized physicochemical features.
- Evaluation: cross-validation plus held-out test set.

## Supervised result
Best model by cross-validation RMSE: **{best_model_name}**

Held-out test result for the best RMSE model:
- MAE: **{best_test["mae"]:.4f}**
- RMSE: **{best_test["rmse"]:.4f}**
- R²: **{best_test["r2"]:.4f}**

Top permutation-importance features for the selected tree-based model:
{chr(10).join(f"- {f}" for f in top_features)}

## Unsupervised result
Selected K-Means cluster count: **{best_k}**

The cluster analysis uses physicochemical variables only and excludes the quality target from clustering.

## Recommendations
1. Use the best validated regression model as a screening/support tool rather than a standalone quality decision.
2. Focus future measurement and feature-engineering work on the strongest predictive variables.
3. Use cluster profiles to identify chemically similar wine groups for further sensory or production analysis.
4. Validate the pipeline on future production batches before operational use.
5. Monitor model drift because relationships between chemistry and quality may change across suppliers, seasons, or production processes.

## Limitations
- Quality labels are subjective/ordinal and may contain evaluator variability.
- Correlated physicochemical variables can make feature-importance interpretation unstable.
- K-Means assumes distance-based, relatively compact groups.
- External validation is required before deployment.

"""
    Path(path).write_text(summary, encoding="utf-8")
