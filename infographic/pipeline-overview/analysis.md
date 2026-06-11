---
title: "Real-Time Credit Card Fraud Detection Pipeline"
topic: "technical"
data_type: "process, comparison, system, data"
complexity: "complex"
point_count: 9
source_language: "en"
user_language: "en"
---

## Main Topic
An enterprise-grade, containerized ML pipeline for real-time credit card fraud detection, built on the Kaggle ULB dataset with strictly chronological splits, latency-constrained LightGBM optimization, probability calibration, and rigorous statistical validation.

## Learning Objectives
After viewing this infographic, the viewer should understand:
1. The end-to-end pipeline architecture from raw transactions to deployment
2. The rigorous methodology that prevents data leakage (temporal splits + train-only resampling)
3. The performance trade-offs and business impact of the calibrated/optimized model vs baseline

## Target Audience
- Knowledge Level: Intermediate to Expert (data scientists, ML engineers, DevOps)
- Context: Explaining the system's architecture, methodology rigor, and performance to stakeholders
- Expectations: Quick visual comprehension of a complex multi-step pipeline; evidence of production readiness

## Content Type Analysis
- Data Structure: Sequential process with embedded comparisons, layered technical details, and statistical validation
- Key Relationships: Pipeline stages depend on prior stages; metrics trade-offs (precision vs latency); rigorous vs leaky literature comparison
- Visual Opportunities: Flow diagram for pipeline, side-by-side metric comparison cards, feature family breakdown, statistical validation callouts

## Key Data Points (Verbatim)
- "284,807 transactions" by European cardholders, "0.172% fraud rate"
- "F1-Score: 0.8041" (maintains baseline, near >0.85 target)
- "Precision: 0.8667" (maintains baseline, near >0.90 target)
- "Recall: 0.7500" (maintains baseline, near >0.80 target)
- "PR-AUC: 0.7672" (+0.0291 improvement from calibration)
- "ROC AUC: 0.9838"
- "Mean Latency: 1.02 ms", "95th Percentile Latency: 1.41 ms" (passes <10ms SLA)
- "Bootstrap 95% CI: [0.7073, 0.8833]", "p-value: 0.9565"
- "Statistical power: 24.8%" at current N_fraud=52, needs "N_fraud=325" for 80% power
- "72 total features" across 4 families: Temporal, Rolling, Interactions, Anomalies
- "1:5" fraud to legitimate ratio via "SMOTE + Random Under-Sampling"
- Decision threshold: "0.7600"
- "177 paths across 27 files" refactored to relative paths

## Layout x Style Signals
- Content type: Technical system + process + comparison -> suggests comparison-matrix, bento-grid, structural-breakdown
- Tone: Rigorous, production-focused, evidence-based -> suggests corporate-memphis, technical-schematic
- Audience: Technical stakeholders -> corporate-memphis for clean data presentation
- Complexity: Complex (9+ points, multi-dimensional) -> comparison-matrix for multi-factor layout

## Recommended Combinations
1. comparison-matrix + corporate-memphis (Recommended): Multi-factor comparison of Baseline vs Calibrated/Optimized across metrics, methodology, and business impact; clean flat vector style suits enterprise/production context.
2. bento-grid + corporate-memphis: Modular layout showing pipeline stages, metrics, and validation as independent cards with clear visual hierarchy.
3. structural-breakdown + technical-schematic: Exploded-view style for pipeline architecture and feature engineering families with blueprint precision.
