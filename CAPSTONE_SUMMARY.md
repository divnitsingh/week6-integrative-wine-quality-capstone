# Week 6 Capstone Summary

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
Best model by cross-validation RMSE: **Random Forest**

Held-out test result for the best RMSE model:
- MAE: **0.4654**
- RMSE: **0.6171**
- R²: **0.4625**

Top permutation-importance features for the selected tree-based model:
- alcohol
- sulphates
- volatile_acidity
- total_sulfur_dioxide
- free_sulfur_dioxide

## Unsupervised result
Selected K-Means cluster count: **2**

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

