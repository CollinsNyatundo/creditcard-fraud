# Real-Time Credit Card Fraud Detection - Structured Infographic Content

## Title
Real-Time Credit Card Fraud Detection Pipeline

## Learning Objectives
1. Understand the end-to-end pipeline architecture from raw transactions to deployment
2. See the rigorous methodology that prevents data leakage (temporal splits + train-only resampling)
3. Compare baseline vs calibrated/optimized model performance and understand the precision-latency trade-off

## Section 1: Problem & Scale
- Key Concept: Extreme class imbalance and real-time latency constraints
- Content: 284,807 transactions | 492 frauds (0.172%) | European cardholders | September 2013
- Visual Element: Scale visualization showing tiny fraud fraction vs total transactions
- Text Labels: "0.172% Fraud Rate" "284,807 Total Transactions"

## Section 2: Pipeline Flow (11 Stages)
- Key Concept: Modular architecture from raw data to production serving
- Content:
  1. Raw Transactions (creditcard.csv)
  2. Data Quality & Exploration (data_exploration.py)
  3. Preprocessing & Robust Scaling (RobustScaler: median + IQR)
  4. Temporal Train/Val/Test Split (60/20/20 chronological)
  5. Class Imbalance: SMOTE + Random Under-Sampling (1:5 ratio, train-only)
  6. Advanced Feature Engineering (72 features x 4 families)
  7. Optuna Hyperparameter Tuning (latency-constrained <8ms pruning)
  8. Model Serialization (optimized_lightgbm.pkl + feature_list.json)
  9. Probability Calibration (calibrate_probabilities.py)
  10. Threshold Optimization (optimize_threshold_pareto.py -> optimal_threshold_v2.json / DB sync)
  11. End-to-End Validation (1000 inference benchmark)
- Visual Element: Linear flow diagram with numbered stages
- Text Labels: Stage numbers, key file names, processing actions

## Section 3: Feature Engineering - 72 Features in 4 Families
- Key Concept: Rich feature set engineered for fraud signal amplification
- Content:
  - A. Temporal: Cyclical hour encoding (sin/cos), Night flag (11PM-5AM), Weekend flag, Velocity tracking
  - B. Rolling Behavior: Mean/std over windows 3/5/10, Z-score deviation
  - C. Interaction: PCA component crosses (Amount*V1, V4, V7), Squared features
  - D. Spending Anomalies: Expanding cumulative average, Global Z-score
- Visual Element: Four categorized blocks/pillars
- Text Labels: Family names, feature counts, key techniques

## Section 4: Baseline vs Calibrated/Optimized - Multi-Metric Comparison
- Key Concept: Performance profile with calibration and latency improvements
- Content:
  - F1-Score: 0.8041 -> 0.8041 (maintains baseline; NEAR TARGET >0.85)
  - Precision: 0.8667 -> 0.8667 (maintains baseline; NEAR TARGET >0.90)
  - Recall: 0.7500 -> 0.7500 (maintains baseline; NEAR TARGET >0.80)
  - PR-AUC (AUPRC): 0.7381 -> 0.7672 (+0.0291 IMPROVEMENT)
  - ROC AUC: 0.9748 -> 0.9838 (EXCELLENT)
  - Mean Latency: 1.40 ms -> 1.02 ms (PASS)
  - 95th Percentile Latency: 3.63 ms -> 1.41 ms (PASS <10ms SLA)
  - 99th Percentile Latency: 7.55 ms -> 1.79 ms (PASS)
- Visual Element: Side-by-side comparison bars or cards with PASS/NEAR TARGET badges
- Text Labels: Metric names, exact values, target thresholds, status badges

## Section 5: Statistical Rigor & Validation
- Key Concept: Small test fraud sample limits statistical significance
- Content:
  - Bootstrap F1: 95% CI [0.7073, 0.8833], median 0.8041
  - Bootstrap Recall: 95% CI [0.6250, 0.8628], mean 0.7488
  - Hypothesis test p-value: 0.9565 (not significant at alpha=0.05)
  - Statistical power: 24.8% at N_fraud=52
  - Need N_fraud=325 (188,953 total transactions) for 80% power
  - Achieves p<0.05 at N_fraud=150, p<0.01 at N_fraud=325
- Visual Element: Callout boxes or mini charts showing CI range and power requirements
- Text Labels: Exact CI bounds, p-values, sample size requirements

## Section 6: Literature Audit - Rigor vs Leakage
- Key Concept: Most published models inflate metrics via global preprocessing/resampling
- Content:
  - Ours: Strict chronological split + train-only resampling -> F1 0.8041 (NO LEAKAGE)
  - 35+ audited papers: Global SMOTE/RUS before split -> inflated F1 0.91-0.999
  - Key differentiator: Temporal data isolation guarantees production generalization
- Visual Element: Split comparison or iceberg metaphor (surface metrics vs hidden leakage)
- Text Labels: "Rigorous" vs "Leaky" labels, F1 ranges, methodology notes

## Section 7: Business Impact
- Key Concept: Optimized deployment preserves business KPIs while improving latency
- Content:
  - Precision maintained: 86.67% across baseline and calibrated/optimized model
  - Recall maintained: 75.00% fraud detection rate
  - Latency improved: 1.41 ms 95th percentile latency and 0.95 ms median latency
  - PR-AUC improved: +0.0291 AUPRC (0.7381 -> 0.7672) from calibration
  - Cost balance: FP = churn risk | FN = chargebacks + fees
- Visual Element: Cost ratio visualization or impact arrows
- Text Labels: Exact percentage values, latency SLA, cost categories

## Section 8: Deployment & MLOps
- Key Concept: Production-ready containerized system with persistent prediction logging
- Content:
  - Docker: python:3.11-slim + libgomp1 + non-root user
  - Docker Compose: Volume mounts for logs/reports persistence
  - Healthcheck: Verifies lightgbm, pandas, sklearn
  - Portability: 177 paths refactored across 27 files to relative paths
  - Cross-platform: Windows Unicode fallback symbols
  - Data path: FastAPI predicts -> Redis WAL -> DLQ worker -> PostgreSQL
- Visual Element: Container/stack diagram with Redis/Postgres flow
- Text Labels: Key Docker components, prediction persistence path, portability notes

## Section 9: Ethical Safeguards
- Key Concept: Responsible AI deployment in financial services
- Content:
  - No fully automated declines without human-in-the-loop
  - Continuous bias monitoring across demographics/geography
  - SHAP values / LightGBM contributions logged for explainability
  - Redress mechanism for cardholders
- Visual Element: Shield or safeguards iconography
- Text Labels: Four key safeguards listed
