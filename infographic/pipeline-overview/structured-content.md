# Real-Time Credit Card Fraud Detection - Structured Infographic Content

## Title
Real-Time Credit Card Fraud Detection Pipeline

## Learning Objectives
1. Understand the end-to-end pipeline architecture from raw transactions to deployment
2. See the rigorous methodology that prevents data leakage (temporal splits + train-only resampling)
3. Compare baseline vs optimized model performance and understand the precision-latency trade-off

## Section 1: Problem & Scale
- Key Concept: Extreme class imbalance and real-time latency constraints
- Content: 284,807 transactions | 492 frauds (0.172%) | European cardholders | September 2013
- Visual Element: Scale visualization showing tiny fraud fraction vs total transactions
- Text Labels: "0.172% Fraud Rate" "284,807 Total Transactions"

## Section 2: Pipeline Flow (9 Stages)
- Key Concept: Modular architecture from raw data to production serving
- Content:
  1. Raw Transactions (creditcard.csv)
  2. Data Quality & Exploration (data_exploration.py)
  3. Preprocessing & Robust Scaling (RobustScaler: median + IQR)
  4. Temporal Train/Val/Test Split (60/20/20 chronological)
  5. Class Imbalance: SMOTE + Random Under-Sampling (1:5 ratio, train-only)
  6. Advanced Feature Engineering (72 features × 4 families)
  7. Optuna Hyperparameter Tuning (latency-constrained <8ms pruning)
  8. Model Serialization (optimized_lightgbm.pkl + feature_list.json)
  9. End-to-End Validation (1000 inference benchmark)
- Visual Element: Linear flow diagram with numbered stages
- Text Labels: Stage numbers, key file names, processing actions

## Section 3: Feature Engineering - 72 Features in 4 Families
- Key Concept: Rich feature set engineered for fraud signal amplification
- Content:
  - A. Temporal: Cyclical hour encoding (sin/cos), Night flag (11PM-5AM), Weekend flag, Velocity tracking
  - B. Rolling Behavior: Mean/std over windows 3/5/10, Z-score deviation
  - C. Interaction: PCA component crosses (Amount×V1, V4, V7), Squared features
  - D. Spending Anomalies: Expanding cumulative average, Global Z-score
- Visual Element: Four categorized blocks/pillars
- Text Labels: Family names, feature counts, key techniques

## Section 4: Baseline vs Optimized - Multi-Metric Comparison
- Key Concept: Performance improvements with controlled latency trade-off
- Content:
  - F1-Score: 0.8041 → 0.8478 (NEAR TARGET >0.85)
  - Precision: 0.8667 → 0.9750 (PASS >0.90, ~80% fewer false declines)
  - Recall: 0.7500 → 0.7500 (MAINTAINED)
  - ROC AUC: 0.9748 → 0.9739 (EXCELLENT)
  - Mean Latency: 1.40ms → 3.03ms
  - 95th Percentile Latency: 3.63ms → 8.89ms (PASS <10ms SLA)
  - 99th Percentile Latency: ~5.20ms → 13.91ms
- Visual Element: Side-by-side comparison bars or cards with PASS/NEAR TARGET badges
- Text Labels: Metric names, exact values, target thresholds, status badges

## Section 5: Statistical Rigor & Validation
- Key Concept: Small test fraud sample limits statistical significance
- Content:
  - Bootstrap F1: 95% CI [0.7593, 0.9195], median 0.8478
  - Hypothesis test p-value: 0.1501 (not significant at α=0.05)
  - Statistical power: 24.8% at N_fraud=52
  - Need N_fraud=325 (188,953 total transactions) for 80% power
  - Achieves p<0.05 at N_fraud=150, p<0.01 at N_fraud=325
- Visual Element: Callout boxes or mini charts showing power curve and CI range
- Text Labels: Exact CI bounds, p-values, sample size requirements

## Section 6: Literature Audit - Rigor vs Leakage
- Key Concept: Most published models inflate metrics via global preprocessing/resampling
- Content:
  - Ours: Strict chronological split + train-only resampling → F1 0.8478 (NO LEAKAGE)
  - 35+ audited papers: Global SMOTE/RUS before split → inflated F1 0.91-0.999
  - Key differentiator: Temporal data isolation guarantees production generalization
- Visual Element: Split comparison or iceberg metaphor (surface metrics vs hidden leakage)
- Text Labels: "Rigorous" vs "Leaky" labels, F1 ranges, methodology notes

## Section 7: Business Impact
- Key Concept: Precision-optimized model protects revenue and customer trust
- Content:
  - Precision 86.67% → 97.50% = ~80% fewer false declines
  - Maintains 75.00% fraud detection rate
  - Latency trade-off viable: 8.89ms safely <10ms bypass threshold
  - Cost balance: FP = churn risk | FN = chargebacks + fees
- Visual Element: Cost ratio visualization or impact arrows
- Text Labels: Exact percentage improvements, latency SLA, cost categories

## Section 8: Deployment & MLOps
- Key Concept: Production-ready containerized system
- Content:
  - Docker: python:3.11-slim + libgomp1 + non-root user
  - Docker Compose: Volume mounts for logs/reports persistence
  - Healthcheck: Verifies lightgbm, pandas, sklearn
  - Portability: 177 paths refactored across 27 files to relative paths
  - Cross-platform: Windows Unicode fallback symbols
- Visual Element: Container/stack diagram
- Text Labels: Key Docker components, portability notes

## Section 9: Ethical Safeguards
- Key Concept: Responsible AI deployment in financial services
- Content:
  - No fully automated declines without human-in-the-loop
  - Continuous bias monitoring across demographics/geography
  - SHAP values / LightGBM contributions logged for explainability
  - Redress mechanism for cardholders
- Visual Element: Shield or safeguards iconography
- Text Labels: Four key safeguards listed
