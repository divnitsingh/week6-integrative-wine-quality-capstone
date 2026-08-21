from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import KFold

from capstone import (
    FEATURES,
    build_models,
    clean_wine_data,
    eda_summary,
    fit_and_evaluate_models,
    kmeans_analysis,
    load_wine_csv,
    prepare_xy,
    regression_cv,
)


SAMPLE = ROOT / "data" / "sample" / "winequality-red-sample.csv"


def test_cleaning_and_schema():
    raw = load_wine_csv(SAMPLE)
    cleaned, report = clean_wine_data(raw)
    assert len(cleaned) > 0
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0
    assert set(FEATURES + ["quality"]).issubset(cleaned.columns)
    assert report["rows_after"] == len(cleaned)


def test_prepare_xy():
    cleaned, _ = clean_wine_data(load_wine_csv(SAMPLE))
    X, y = prepare_xy(cleaned)
    assert X.shape[1] == len(FEATURES)
    assert len(X) == len(y)


def test_eda_summary():
    cleaned, _ = clean_wine_data(load_wine_csv(SAMPLE))
    summary = eda_summary(cleaned)
    assert "feature" in summary.columns
    assert len(summary) == len(FEATURES) + 1


def test_regression_cv_runs():
    cleaned, _ = clean_wine_data(load_wine_csv(SAMPLE))
    X, y = prepare_xy(cleaned)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    results = regression_cv(build_models(), X, y, cv)
    assert len(results) == 4
    assert {"model", "mae_mean", "rmse_mean", "r2_mean"} <= set(results.columns)


def test_regression_test_evaluation():
    cleaned, _ = clean_wine_data(load_wine_csv(SAMPLE))
    X, y = prepare_xy(cleaned)
    X_train, X_test = X.iloc[:45], X.iloc[45:]
    y_train, y_test = y.iloc[:45], y.iloc[45:]
    results, fitted = fit_and_evaluate_models(
        build_models(), X_train, y_train, X_test, y_test
    )
    assert len(results) == 4
    assert np.isfinite(results["rmse"]).all()
    assert len(fitted) == 4


def test_kmeans_runs():
    cleaned, _ = clean_wine_data(load_wine_csv(SAMPLE))
    X, _ = prepare_xy(cleaned)
    metrics, best_k, model, scaler, profile, sizes = kmeans_analysis(
        X, ROOT / "outputs" / "figures"
    )
    assert 2 <= best_k <= 6
    assert len(metrics) == 5
    assert profile.shape[0] == best_k
    assert sizes["count"].sum() == len(X)
