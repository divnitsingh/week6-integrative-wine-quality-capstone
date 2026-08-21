# Week 6 — Integrative Data Science Capstone

A GitHub-ready, hands-on capstone that combines the complete data-science workflow learned during Weeks 1–5.

## Capstone question

**Can physicochemical measurements be used to predict wine quality, and can the same measurements reveal meaningful segments of wines without using the quality score?**

This project combines:

1. Data acquisition from a public source.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Supervised learning — regression.
5. Unsupervised learning — clustering.
6. Model evaluation and comparison.
7. Interpretation, recommendations, and reflection.

## Dataset

**Dataset:** Wine Quality — Red Wine  
**Provider:** UCI Machine Learning Repository  
**Official page:**  
https://archive.ics.uci.edu/dataset/186/wine+quality

**Official red-wine CSV:**  
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

The dataset contains physicochemical measurements of red wine samples and an integer quality score.

The project downloads the public CSV in normal mode. A compact bundled fixture is included for offline testing and smoke runs; the loader accepts both the official semicolon-delimited UCI format and the comma-delimited fixture.

## Business/research framing

A winery or quality-control team could use this workflow to:

- estimate expected quality from measurable chemistry,
- identify which chemical variables are most useful for prediction,
- compare model performance before operationalizing a predictive system,
- and segment wines into chemically similar groups for exploratory profiling.

The project deliberately treats model output as decision support, not a replacement for sensory evaluation or human quality-control procedures.

## Project structure

```text
week6_integrative_wine_capstone/
├── README.md
├── CAPSTONE_SUMMARY.md
├── DATA_ATTRIBUTION.md
├── requirements.txt
├── run_capstone.py
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
│   └── week6_integrative_capstone.ipynb
├── src/
│   └── capstone.py
├── tests/
│   └── test_capstone.py
└── outputs/
    ├── figures/
    ├── tables/
    └── models/
```

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the capstone

### Full project
```bash
python run_capstone.py
```

### Offline smoke test
```bash
python run_capstone.py --sample
```

Normal mode downloads the official UCI red-wine CSV. If the network is unavailable, the runner automatically falls back to the bundled sample fixture.

## Analytical workflow

### Phase 1 — Acquisition
- Download the current public UCI dataset.
- Record source metadata.
- Preserve raw and processed layers separately.

### Phase 2 — Cleaning
- Convert all variables to numeric form.
- Drop invalid rows created by failed conversions.
- Remove duplicate records.
- Validate plausible ranges.
- Confirm no missing values remain in the modeling frame.

### Phase 3 — EDA
- Descriptive statistics.
- Quality-score distribution.
- Feature distributions.
- Correlation heatmap.
- Quality-by-alcohol analysis.
- Aggregated mean quality by quality score.

### Phase 4 — Supervised learning
**Target:** wine quality score.

Models:
- Baseline DummyRegressor.
- Random Forest Regressor.
- Gradient Boosting Regressor.

Validation:
- 5-fold KFold cross-validation.
- Metrics: MAE, RMSE, R².
- Held-out test-set evaluation.

The strongest model is selected by cross-validation RMSE.

### Phase 5 — Unsupervised learning
K-Means clustering uses physicochemical features only and **does not include the quality target**.

- StandardScaler.
- Candidate k values 2–6.
- Silhouette score for model selection.
- PCA projection for visualization.
- Cluster-size summary.
- Standardized cluster profile.

### Phase 6 — Evaluation and interpretation
- Compare models on validation and test metrics.
- Review feature importances from the tree-based regressor.
- Examine cluster characteristics.
- Connect findings to practical recommendations.
- Discuss limitations and next steps.

## Outputs

### Processed datasets
- `data/processed/cleaned_wine_quality.csv`
- `data/processed/model_predictions.csv`
- `data/processed/cluster_assignments.csv`
- `data/processed/feature_importance.csv`
- `data/processed/capstone_metadata.json`

### Tables
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/eda_summary.csv`
- `outputs/tables/cv_model_results.csv`
- `outputs/tables/test_model_results.csv`
- `outputs/tables/kmeans_metrics.csv`
- `outputs/tables/cluster_profile.csv`
- `outputs/tables/cluster_sizes.csv`

### Figures
- `01_quality_distribution.png`
- `02_feature_distributions.png`
- `03_correlation_heatmap.png`
- `04_quality_vs_alcohol.png`
- `05_model_comparison.png`
- `06_feature_importance.png`
- `07_kmeans_silhouette.png`
- `08_kmeans_pca.png`
- `09_cluster_profile_heatmap.png`

### Saved models
- `outputs/models/random_forest_regressor.joblib`
- `outputs/models/gradient_boosting_regressor.joblib`
- `outputs/models/kmeans.joblib`
- `outputs/models/scaler.joblib`

## Running tests

```bash
python -m pytest -q
```

The automated tests cover:
- data loading and cleaning,
- feature/target separation,
- supervised model fitting,
- metric generation,
- clustering outputs,
- and end-to-end smoke behavior.

## Notebook

Open:

```bash
jupyter notebook notebooks/week6_integrative_capstone.ipynb
```

The notebook presents the complete workflow in narrative form and generates the same artifacts as the script.

## Important interpretation notes

- Regression metrics describe predictive performance; they do not prove causality.
- Feature importance can be affected by correlated predictors.
- K-Means clusters are similarity segments, not automatically meaningful business categories.
- The quality score is an expert-derived target, but the project does not treat it as a perfect ground truth.
- Real production use would require external validation, monitoring, and domain review.

## GitHub upload

Suggested repository:

`week6-integrative-wine-quality-capstone`

Upload the **contents of the extracted project folder** to GitHub, then use the repository URL in the internship certificate submission field.

## Attribution

See `DATA_ATTRIBUTION.md`.
