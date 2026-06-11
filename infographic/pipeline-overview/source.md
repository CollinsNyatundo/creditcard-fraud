# Real-Time Credit Card Fraud Detection Pipeline - Source Content for Infographic

## Project Overview
An enterprise-grade, containerized machine learning pipeline designed to identify fraudulent transactions in real-time under strict latency constraints (<10ms 95th percentile). Built on the Kaggle Credit Card Fraud Detection Dataset (284,807 transactions, 0.172% fraud rate, European cardholders, September 2013, published by ULB Machine Learning Group).

## Key Performance Metrics (Verified on test dataset)

| Metric | Project Target | Baseline Model | Calibrated + Optimized Model | Status |
|--------|---------------|---------------|-------------------------|--------|
| F1-Score | > 0.85 | 0.8041 | 0.8041 | NEAR TARGET |
| Precision | > 0.90 | 0.8667 | 0.8667 | NEAR TARGET |
| Recall | > 0.80 | 0.7500 | 0.7500 | NEAR TARGET |
| PR-AUC (AUPRC) | N/A | 0.7381 | 0.7672 | +0.0291 IMPROVEMENT |
| ROC AUC | N/A | 0.9748 | 0.9838 | EXCELLENT |
| Mean Latency | N/A | 1.40 ms | 1.02 ms | PASS |
| 95th Percentile Latency | < 10.00 ms | 3.63 ms | 1.41 ms | PASS |
| 99th Percentile Latency | N/A | 7.55 ms | 1.79 ms | PASS |

## Statistical Validation
- Bootstrap F1-Score: 95% CI [0.7073, 0.8833], median 0.8041
- Bootstrap Recall 95% CI: [0.6250, 0.8628], mean 0.7488
- Hypothesis test p-value: 0.9565 (not statistically significant due to small test fraud sample: 52 cases)
- Statistical power at current scale: 24.8% (needs 325 fraud cases for 80% power)
- Hypothetical p-value scaling: achieves significance (p<0.05) at N_fraud=150, highly significant (p<0.01) at N_fraud=325
- Total dataset needed for 80% power under 60/20/20 chronological split with natural 0.172% fraud rate: >944,000 transactions

## Pipeline Architecture (Modular Flow)
1. Raw Transactions: creditcard.csv (284,807 transactions)
2. Data Quality & Exploration: data_exploration.py - EDA, missing value checks, EDA reports
3. Preprocessing & Robust Scaling: feature_engineering.py - RobustScaler (median/IQR)
4. Temporal Train/Val/Test Split: Chronological cutoff (60/20/20) - simulates real-world deployment
5. Class Imbalance Solver: SMOTE + Random Under-Sampling -> 1:5 fraud:legitimate ratio
6. Advanced Feature Engineering: 72 total features across 4 families
7. Optuna Hyperparameter Tuning: Latency-constrained (<8ms 95th percentile pruning)
8. Model Serialization: optimized_lightgbm.pkl optimized via latency-constrained Optuna search
9. Probability Calibration: calibrate_probabilities.py -> Platt-scaled calibrated model wrapper
10. Threshold Optimization: optimize_threshold_pareto.py -> optimal_threshold_v2.json/DB sync
11. End-to-End Validation: 1000 single-transaction inference benchmark

## Feature Engineering - 72 Features in 4 Families

### A. Temporal Features
- Cyclical hour encoding (sin/cos) - preserves temporal adjacency (hour 23 approx hour 0)
- Night flag (11 PM - 5 AM)
- Weekend flag
- Time since last transaction (velocity tracking)

### B. Rolling Behavior Statistics (windows: 3, 5, 10)
- Rolling mean & standard deviation
- Z-score deviation against recent history

### C. Interaction Features
- PCA component crosses (Amount * V1, V4, V7)
- Squared features for non-linear deviations

### D. Spending Anomalies
- Expanding cumulative average
- Global Z-score on overall dataset

## Resampling Strategy
- SMOTE: Synthetic fraud generation via k-nearest neighbors
- Random Under-Sampling: Discard majority class samples
- Ratio: 1:5 (fraud:legitimate) in training only
- Validation/test retain original 0.172% distribution (no leakage)

## Hyperparameter Tuning (Optuna)
- Objective: Maximize validation F1-score
- Pruning: Trials exceeding 8ms 95th percentile latency are pruned
- Search space: num_leaves (20-100), max_depth (3-10), learning_rate (0.01-0.1), min_child_samples (10-100), subsample (0.6-1.0), colsample_bytree (0.6-1.0), reg_alpha (0-10), reg_lambda (0-10)

## Model Inference
- Decision threshold: 0.7600 (calibrated for F1 optimization / cost-aware Pareto sweep)
- 72 features recreated identically during inference
- LightGBM tree structure for fast inference
- Platt scaling improves probability calibration

## Literature Audit - Rigorous vs Leaky Methods
- Our pipeline: Strict chronological split, resampling only on train fold -> F1 0.8041 (NO LEAKAGE)
- 35+ papers in audit: Most apply SMOTE globally before train/test split -> inflated F1 (0.91-0.99)
- Key differentiator: Temporal data isolation guarantees production generalization

## Business Impact
- Precision maintained: 86.67% stable across baseline and calibrated/optimized model
- Recall maintained: 75.00% fraud detection rate
- Latency improved under optimization: 1.41 ms 95th percentile latency and 0.95 ms median latency, still safely <10ms SLA
- Cost ratio: FP (churn risk) vs FN (fraud loss + chargebacks)
- PR-AUC improved: +0.0291 AUPRC with calibration (0.7381 -> 0.7672)

## Deployment & MLOps
- Docker: python:3.11-slim + libgomp1 for LightGBM
- Docker Compose: Volume mounts for persistent logs/reports
- Healthcheck: Verifies core ML libraries
- Non-root user (appuser:appgroup)
- Relative paths (177 paths refactored across 27 files)
- Windows Unicode compatibility (fallback symbols)
- Write-Ahead Logging on Redis for prediction persistence to PostgreSQL

## Ethical Safeguards
- No fully automated declines without human-in-the-loop
- Continuous bias monitoring across demographics/geography
- SHAP values / LightGBM contributions logged for explainability
- Redress mechanism for cardholders
