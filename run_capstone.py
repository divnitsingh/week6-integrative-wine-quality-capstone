from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from capstone import (
    FEATURES,
    TARGET,
    UCI_WINE_URL,
    build_models,
    clean_wine_data,
    download_wine_quality,
    eda_summary,
    feature_importance_table,
    fit_and_evaluate_models,
    kmeans_analysis,
    load_wine_csv,
    plot_feature_importance,
    plot_model_comparison,
    prepare_xy,
    regression_cv,
    run_eda_figures,
    save_json,
    write_capstone_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 6 integrative capstone.")
    parser.add_argument("--sample", action="store_true", help="Use bundled smoke-test fixture.")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    figures_dir = ROOT / "outputs" / "figures"
    tables_dir = ROOT / "outputs" / "tables"
    models_dir = ROOT / "outputs" / "models"

    sample_path = ROOT / "data" / "sample" / "winequality-red-sample.csv"
    raw_path = raw_dir / "winequality-red.csv"

    try:
        if args.sample:
            df_raw = load_wine_csv(sample_path)
            mode = "sample"
        else:
            raw_dir.mkdir(parents=True, exist_ok=True)
            if not raw_path.exists():
                download_wine_quality(raw_path)
            df_raw = load_wine_csv(raw_path)
            mode = "official UCI"
    except Exception as exc:
        print(f"Live data acquisition unavailable ({exc}); using bundled sample.")
        df_raw = load_wine_csv(sample_path)
        mode = "sample-fallback"

    cleaned, quality_report = clean_wine_data(df_raw)
    X, y = prepare_xy(cleaned)

    # EDA artifacts
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(processed_dir/"cleaned_wine_quality.csv", index=False)
    pd.DataFrame([quality_report]).to_csv(tables_dir/"data_quality_summary.csv", index=False)
    eda_summary(cleaned).to_csv(tables_dir/"eda_summary.csv", index=False)
    run_eda_figures(cleaned, figures_dir)

    # Supervised split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    cv_results = regression_cv(models, X_train, y_train, cv)
    test_results, fitted = fit_and_evaluate_models(
        models, X_train, y_train, X_test, y_test
    )

    best_model_name = cv_results.sort_values("rmse_mean").iloc[0]["model"]
    best_model = fitted[best_model_name]
    predictions = best_model.predict(X_test)

    prediction_frame = X_test.copy()
    prediction_frame["actual_quality"] = y_test.to_numpy()
    prediction_frame["predicted_quality"] = predictions
    prediction_frame["absolute_error"] = (y_test.to_numpy() - predictions).astype(float).round(6)
    prediction_frame.to_csv(processed_dir/"model_predictions.csv", index=False)

    cv_results.to_csv(tables_dir/"cv_model_results.csv", index=False)
    test_results.to_csv(tables_dir/"test_model_results.csv", index=False)

    # Feature importance if tree-based best model; otherwise use best Random Forest as
    # a dedicated interpretable importance model.
    importance_model = best_model
    if best_model_name not in {"Random Forest", "Gradient Boosting"}:
        importance_model = fitted["Random Forest"]

    importance = feature_importance_table(importance_model, X_test, y_test)
    importance.to_csv(processed_dir/"feature_importance.csv", index=False)
    plot_model_comparison(test_results, figures_dir/"05_model_comparison.png")
    plot_feature_importance(importance, figures_dir/"06_feature_importance.png")

    # Save models.
    for name in ["Random Forest", "Gradient Boosting"]:
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(fitted[name], models_dir/f"{safe_name}_regressor.joblib")

    # Unsupervised clustering.
    k_metrics, best_k, kmeans_model, scaler, profile, sizes = kmeans_analysis(
        X,
        figures_dir,
        random_state=42,
    )
    k_metrics.to_csv(tables_dir/"kmeans_metrics.csv", index=False)
    profile.to_csv(tables_dir/"cluster_profile.csv")
    sizes.to_csv(tables_dir/"cluster_sizes.csv", index=False)

    scaler_path = models_dir/"scaler.joblib"
    kmeans_path = models_dir/"kmeans.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans_model, kmeans_path)

    cluster_labels = kmeans_model.predict(scaler.transform(X))
    cluster_assignments = X.copy()
    cluster_assignments["cluster"] = cluster_labels
    cluster_assignments["quality"] = y.to_numpy()
    cluster_assignments.to_csv(processed_dir/"cluster_assignments.csv", index=False)

    metadata = {
        "dataset": "UCI Wine Quality — Red Wine",
        "source_url": UCI_WINE_URL,
        "acquisition_mode": mode,
        "rows_raw": int(len(df_raw)),
        "rows_clean": int(len(cleaned)),
        "features": FEATURES,
        "target": TARGET,
        "best_regression_model": best_model_name,
        "best_cv_rmse": float(cv_results.sort_values("rmse_mean").iloc[0]["rmse_mean"]),
        "best_test_rmse": float(test_results.sort_values("rmse").iloc[0]["rmse"]),
        "best_test_mae": float(test_results.sort_values("rmse").iloc[0]["mae"]),
        "best_test_r2": float(test_results.sort_values("rmse").iloc[0]["r2"]),
        "selected_k": int(best_k),
        "quality_report": quality_report,
    }
    save_json(metadata, processed_dir/"capstone_metadata.json")

    write_capstone_summary(
        ROOT/"CAPSTONE_SUMMARY.md",
        metadata,
        cv_results,
        test_results,
        best_model_name,
        best_k,
        importance,
    )

    print("\nWeek 6 capstone complete.")
    print("Mode:", mode)
    print("Rows:", len(cleaned))
    print("Best regression model:", best_model_name)
    print("Best CV RMSE:", f"{metadata['best_cv_rmse']:.4f}")
    print("Best test RMSE:", f"{metadata['best_test_rmse']:.4f}")
    print("Selected K-Means k:", best_k)
    print("Outputs written to data/processed, outputs/, and CAPSTONE_SUMMARY.md")


if __name__ == "__main__":
    main()
