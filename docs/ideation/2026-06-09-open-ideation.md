---
date: 2026-06-09
topic: open-ideation
focus: surprise-me mode
mode: repo-grounded
---

# Ideation: Credit Card Fraud Detection Optimizations

## Grounding Context
The system is a credit card fraud detection FastAPI backend with a Next.js frontend, backed by PostgreSQL and Redis. Major system characteristics:
- Loads the LightGBM booster once at startup via lifecycle manager.
- Preprocesses transaction amounts, computes rolling windows, and calls KMeans behavior clustering.
- Employs background tasks for asynchronous TreeSHAP explainability.
- Features real-time streaming to the dashboard over WebSockets.
- Incorporates SlowAPI for rate limiting (100/sec).
- Leverages Evidently for drift monitoring and custom scripts for bootstrap validation.

## Topic Axes
Decomposition skipped — surprise-me mode

## Ranked Ideas

### 1. Streaming Feature Store Increments in Redis
**Description:** Recalculating rolling windows on transaction histories dynamically on the fly slows down predict requests. Implementing an incremental update approach using Redis Sorted Sets and Welford's algorithm guarantees O(1) feature reconstruction latency.
**Basis:** `direct: app/routes/predict.py L140-160` (dynamic rolling window aggregations).
**Rationale:** Keeps the transaction route fast (<10ms) even for users with thousands of transaction history records.
**Downsides:** Introduces state tracking complexity and potential out-of-sync risks if updates fail.
**Confidence:** 90%
**Complexity:** Medium
**Status:** Unexplored

### 2. Unified MLflow Pipeline Serialization
**Description:** Rather than loading raw preprocessor pickled binaries (`RobustScaler`, `KMeans`) and configuration `.json` files from local disk, serialize the entire preprocessing pipeline inside MLflow as a single artifact along with the LightGBM model.
**Basis:** `direct: app/main.py L48-67` (separate file loadings from local assets).
**Rationale:** Removes local path dependencies and guarantees that the active model has matching scaler parameters.
**Downsides:** Requires modifying model deployment CI/CD pipelines.
**Confidence:** 95%
**Complexity:** Low
**Status:** Unexplored

### 3. Cluster-Based Centroid SHAP Approximations
**Description:** Running full TreeSHAP traversals on background tasks for every flagged transaction can exhaust memory/CPU during fraud bursts. Utilizing behavior cluster IDs to return pre-computed centroid SHAP explanations saves considerable computation.
**Basis:** `direct: app/routes/predict.py L410-418` (schedules `compute_and_store_shap` on background tasks).
**Rationale:** Relieves server load and queue congestion during fraud spikes, serving as a smart approximation.
**Downsides:** Approximation is less precise than calculating exact transaction-level SHAP values.
**Confidence:** 80%
**Complexity:** Medium
**Status:** Unexplored

### 4. Resilience Fallback with Local Model Cache
**Description:** Add a local directory fallback `/app/model_cache` containing the last running MLflow model. If FastAPI cannot connect to the MLflow tracker during startup, it serves traffic using this cache instead of launching in a degraded state.
**Basis:** `direct: app/main.py L44` (loads model directly from MLflow tracking server).
**Rationale:** Ensures service high availability (HA) during transient tracking server outages.
**Downsides:** May serve slightly outdated model versions until MLflow recovers.
**Confidence:** 95%
**Complexity:** Low
**Status:** Unexplored

### 5. Event-Driven WebSocket Dwell Buffering
**Description:** Instantly streaming every single transaction event to the WebSocket publisher under high throughput saturates connections. Buffer transaction records on the backend and flush them in micro-batches (e.g. 100ms) to the Next.js client.
**Basis:** `direct: app/routes/predict.py L421` (publishes transactions synchronously).
**Rationale:** Smoothens Next.js UI rendering performance and stabilizes websocket connection pools.
**Downsides:** Introduces a minor delay (up to 100ms) in streaming updates to the dashboard.
**Confidence:** 85%
**Complexity:** Low
**Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Double-Consensus Instant Card Blocking | Requires complex payment processor integrations and risk clearance, expanding beyond standard ML system scope. |
| 2 | In-Memory Hot-Card Cache | Storing in-memory card states across multi-worker uvicorn processes introduces data drift/consistency issues. |
| 3 | DB Index Tuning for Prediction Logs | Too tactical; typical database administration that fails the meeting-test (doesn't warrant major team discussion). |
| 4 | UI Customizer | Below ambition floor; minor dashboard customization that doesn't target fraud detection performance. |
| 5 | Custom Rate Limiter Middleware | SlowAPI is already configured and working; replacing it adds little value. |
| 6 | Additional Unit Tests | Too tactical; standard testing duty that doesn't warrant system design ideation. |
| 7 | Vault Integration | Standard DevSecOps security practice rather than a custom ML system design improvement. |
