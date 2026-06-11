---
date: 2026-06-09
topic: recall-cost-optimization-metrics
---

# Recall and Cost Optimization Metrics Requirements

## Summary

Transition the credit card fraud detection system from a precision-constrained F1-maximizing threshold selection to a recall-centric, business-cost-minimized framework. The core outcome is to implement (1) a dynamic financial utility cost function that finds the cost-minimizing threshold and (2) a recall-constrained threshold targeting Recall ≥ 0.85 — with both thresholds computed and stored so an operator config determines which is active. AUPRC replaces ROC AUC as the primary ranking metric, and G-mean is added as a co-metric alongside F1.

---

## Problem Frame

The current calibrated model uses a threshold of `0.77`, yielding Precision `97.44%` but Recall only `73.08%` — meaning 27% of actual fraud cases are missed. In credit card fraud, a False Negative (missed fraud) generates chargebacks, regulatory penalties, and customer reimbursements: costs that are 3–5x higher per event than a False Positive (customer friction and support call). Optimizing for F1-balance treats precision and recall as equal-weight, which structurally underweights recall against what the business actually loses.

Additionally, the Kaggle dataset authors explicitly recommend AUPRC over ROC AUC for this dataset (0.172% fraud rate). Our current evaluation reports ROC AUC as the primary ranking metric, which is inflated by the overwhelming volume of true negatives and does not reflect model quality on the minority fraud class.

---

## Key Decisions

- **Recall is the primary optimization target.** Minimum acceptable: Recall ≥ 0.85. Ideal target for enterprise fintech: ≥ 0.90. Precision is allowed to drop to 0.85–0.90 to achieve this.
- **Dual-threshold operation.** Two thresholds are computed and stored: `cost_min` (minimizes total dynamic business cost) and `recall_target` (highest-recall threshold satisfying Recall ≥ 0.85). An operator config key in `models/optimal_threshold_v2.json` determines which is active at inference time.
- **Dynamic cost parameters.** $C_{\text{fraud}} = \text{Amount}_i + \$15$ (transaction amount + chargeback fee) per FN. $C_{\text{churn}} = \$50$ flat per FP.
- **AUPRC replaces ROC AUC as the primary ranking metric.** The dataset is imbalanced (0.172% fraud); AUPRC correctly measures ranking quality on the minority class. ROC AUC remains logged for backward compatibility but is secondary.
- **G-mean added as a co-metric alongside F1.** Per arxiv empirical study (2208.11904), combined F1 + G-mean is the strongest evaluation pairing for imbalanced fraud classification.
- **Latency SLA is non-negotiable.** Recall and cost improvements must not push 95th-percentile inference latency above 10ms.

---

## Requirements

### Priority 1 — Recall Optimization

- R1. The threshold search must compute and save a `recall_target` threshold defined as the lowest threshold value that achieves Recall ≥ 0.85 on the validation set.
- R2. If no threshold achieves Recall ≥ 0.85, the recall_target must be set to the threshold that maximizes recall (best effort), and the requirements doc must flag this as an outstanding data constraint.
- R3. The evaluation report must surface Recall as the first-listed metric, with the 0.85 / 0.90 targets explicitly marked as PASS/FAIL.

### Priority 2 — AUPRC as Primary Ranking Metric

- R4. AUPRC (Average Precision / PR-AUC) must be the primary ranking metric in all evaluation outputs, replacing ROC AUC at the top of the metrics table.
- R5. The Precision-Recall curve must be plotted and saved to `reports/` for every evaluation run.
- R6. AUPRC minimum acceptable: > 0.70. Current calibrated baseline: 0.7345 (acceptable — must not regress).
- R7. ROC AUC must remain logged as a secondary metric for backward compatibility.

### Priority 3 — Latency SLA

- R8. 95th-percentile inference latency must remain < 10ms. Current baseline: 1.15ms. Any change to threshold or inference path must be verified against this SLA.

### Priority 4 — F1-Score and G-mean Co-metrics

- R9. F1-score minimum acceptable: > 0.80. Current calibrated baseline: 0.8352 (acceptable). Target: > 0.85.
- R10. G-mean (geometric mean of sensitivity and specificity) must be computed and logged alongside F1 in all evaluation outputs.
- R11. The evaluation report must label F1 as a secondary comparative metric subordinate to Recall and AUPRC in the business priority hierarchy.

### Priority 5 — Dynamic Cost Function

- R12. The threshold sweep must compute total business cost at each candidate threshold:
  - $\text{Total Cost} = \sum_{i \in \text{FN}} (\text{Amount}_i + 15) + \sum_{j \in \text{FP}} 50$
- R13. The cost-minimized threshold (`cost_min`) must be stored alongside `recall_target` in `models/optimal_threshold_v2.json`.
- R14. The active threshold key (either `cost_min` or `recall_target`) must be configurable via `models/optimal_threshold_v2.json` without restarting the API.
- R15. The Cost vs. Threshold curve must be plotted and saved to `reports/` for every optimization run.

### Priority 6 — Monitoring & Drift

- R16. The drift monitor (`utils/drift_monitor.py`) must log AUPRC trend across monitoring windows, not just snapshot ROC AUC.
- R17. Transaction amount distributions must be tracked in the drift monitor to ensure cost function estimates remain valid as spending patterns shift.

---

## Acceptance Examples

- AE1. **Recall constraint satisfied:**
  - **Given**: validation set with 52 actual fraud cases.
  - **When**: `recall_target` threshold search runs.
  - **Then**: a threshold is selected such that at least 44 of 52 fraud cases are caught (Recall ≥ 0.85).

- AE2. **Cost minimization:**
  - **Given**: validation set with actual transaction amounts.
  - **When**: the cost sweep runs at 0.01 threshold steps.
  - **Then**: `cost_min` threshold is the candidate with the lowest value of $\sum_{\text{FN}} (\text{Amount} + 15) + \sum_{\text{FP}} 50$.

- AE3. **Operator config toggle:**
  - **Given**: `models/optimal_threshold_v2.json` has `"active": "recall_target"`.
  - **When**: `/predict` loads threshold at startup.
  - **Then**: inference uses the recall-constrained threshold without API restart.

- AE4. **AUPRC non-regression:**
  - **Given**: calibrated model baseline AUPRC = 0.7345.
  - **When**: any new threshold or calibration is applied.
  - **Then**: AUPRC must be ≥ 0.70 (logged and flagged if it falls below).

---

## Success Criteria

| Priority | Metric | Target | Current | Gate |
|----------|--------|--------|---------|------|
| 1 | Recall | ≥ 0.85 | 0.7308 | FAIL — primary gap |
| 2 | AUPRC | > 0.70 | 0.7345 | PASS — must not regress |
| 3 | Latency p95 | < 10ms | 1.15ms | PASS — must not regress |
| 4 | F1-Score | > 0.80 | 0.8352 | PASS |
| 5 | Total Cost | Minimize | baseline TBD | No regression |

---

## Scope Boundaries

### Deferred for later
- Real-time threshold hot-reloading via Redis (operator config toggle is sufficient for v1; a future API endpoint can expose live updates).
- Training new model variants with a cost-weighted loss function (post-hoc threshold tuning is v1; custom LightGBM objective is v2).
- Acquiring more data (325+ fraud cases needed for 80% statistical power — noted as a data constraint, not addressable within this pipeline change).

### Outside this product's identity
- Directly blocking cards at the payment network level (this service returns probability scores and authorization recommendations only).

---

## Outstanding Questions

- **Dashboard Integration**: The Metabase dashboard will query the `system_config` table and `predictions` table to display the active threshold and the real-time distribution of transaction amounts and probabilities.
- **G-mean Status**: G-mean is net-new code and will be added as a utility helper function in `calibrate_probabilities.py` and `final_model_evaluation.py`.

---

## Dependencies / Assumptions

- The validation set contains transaction `Amount` values required for dynamic cost calculation (confirmed present in `data/processed/val.csv`).
- The test holdout contains 52 fraud cases — achieving Recall ≥ 0.85 requires catching at least 44. At the current model's discrimination ability (ROC AUC 0.9748), this is achievable by lowering the threshold.
- Precision is expected to drop when recall is raised; 0.85–0.90 precision is acceptable.

---

## Sources / Research

- Kaggle creditcardfraud dataset: AUPRC explicitly recommended over ROC AUC for this 0.172% imbalance ratio — [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Recall primacy in fraud detection — [towardsdatascience.com](https://towardsdatascience.com/credit-card-fraud-detection-with-different-sampling-techniques-cece7734acc5/)
- Recall ≥ 0.85 minimum target; G-mean + F1 as best co-metric pairing — [arxiv.org/abs/2208.11904](https://arxiv.org/abs/2208.11904)
- Cost function threshold selection — [cesarsotovalero.net](https://www.cesarsotovalero.net/blog/evaluation-metrics-for-real-time-financial-fraud-detection-ml-models.html)
- 95th-percentile latency SLA — [coralogix.com](https://coralogix.com/ai-blog/how-to-optimize-ml-fraud-detection-a-guide-to-monitoring-performance/)
- Existing pipeline files to modify:
  - `model/src/optimize_threshold_pareto.py` — add dual-threshold output and cost sweep
  - `model/src/calibrate_probabilities.py` — add AUPRC and G-mean logging
  - `model/src/final_model_evaluation.py` — reorder metrics, add PR curve, G-mean, cost table
  - `utils/drift_monitor.py` — add AUPRC trend tracking
  - `models/optimal_threshold_v2.json` — add `cost_min`, `recall_target`, `active` fields
