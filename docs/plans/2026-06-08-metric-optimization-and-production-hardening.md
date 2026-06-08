# Metric Optimization & Production Hardening Plan

This plan details the implementation steps required to improve the classification metrics (F1, Precision, Recall) under strict chronological split constraints and resolve production-level gaps (resiliency, drift detection, and security audibility).

## User Review Required

> [!IMPORTANT]
> **Key Architectural Decisions for Approval:**
> 1. **Focal Loss Integration**: Using custom focal loss in LightGBM to prioritize hard-to-classify fraud cases.
> 2. **Fail-Open Resiliency**: Implementing a fallback routine in the FastAPI endpoint if Redis caching or database queries time out, ensuring the payment authorization SLA is never blocked.
> 3. **Lease-isolated Clustering**: Training behavior proxy clusters (KMeans) strictly on training split data to prevent lookahead leakage.
> 4. **Model Drift Monitoring**: Integrating the `evidently` package to automate statistical data drift and prediction score drift reporting.

---

## Proposed Changes

### Component 1: ML Science & Tuning

#### [MODIFY] [hyperparameter_tuning.py](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/hyperparameter_tuning.py)
- Expand the Optuna tuning space from 50 to 150 trials.
- Implement focal loss training in LightGBM parameters configuration.
- Integrate business utility cost-optimization scoring in the objective function.

#### [NEW] [proxy_clustering.py](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/proxy_clustering.py)
- Implement a behavior proxy clustering preprocessor (using `scikit-learn`'s `KMeans`).
- Train the clusterer strictly on the training partition and save the serialized processor to `models/`.
- Align feature generation to map incoming real-time records to their nearest cluster index.

---

### Component 2: Production Hardening & Resilience

#### [MODIFY] [predict.py](file:///d:/Projects/ai-ml/creditcard-fraud/app/routes/predict.py)
- Wrap all Redis database calls (`push_card_amount`, `get_card_history_with_timestamps`) in robust `try...except` blocks.
- If Redis is down, return default fallback behavior velocities (e.g. median historic values) to guarantee low latency.
- Log prediction metadata, thresholds, probabilities, and top SHAP-based feature importance indicators directly to the PostgreSQL ledger for audit explainability.

#### [NEW] [drift_monitor.py](file:///d:/Projects/ai-ml/creditcard-fraud/utils/drift_monitor.py)
- Implement a script using the `evidently` library to compare incoming live database transaction features against baseline training distributions.
- Perform statistical tests (KS-test, Chi-Square, and PSI) to identify covariate and prediction/target drift.
- Export results to `reports/drift_report.json` and generate an interactive dashboard at `reports/drift_dashboard.html`.

#### [MODIFY] [requirements.txt](file:///d:/Projects/ai-ml/creditcard-fraud/requirements.txt)
- Add `evidently>=0.4.0` to the python dependencies list.

---

## Verification Plan

### Automated Tests
- Run unit tests to check API routing and default value fallbacks:
  ```bash
  .venv\Scripts\pytest tests/unit/test_predict_endpoint.py
  ```
- Verify model feature alignment and inference compatibility.

### Manual Verification
- Stop the Redis service temporarily to trigger mock timeouts, confirming that `/predict` fails open gracefully and returns predictions under 10ms.
