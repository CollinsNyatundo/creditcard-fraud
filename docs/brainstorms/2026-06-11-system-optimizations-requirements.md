---
date: 2026-06-11
topic: system-optimizations
focus: decoupled performance and resilience hardening
mode: repo-grounded
---

# Requirements: Fraud Detection Pipeline Hardening

## Summary
Proposes a decoupled, high-throughput optimization release for the Credit Card Fraud Detection Pipeline. The system offloads Redis feature updates and PostgreSQL logs to separate async queues, processes webhook alerts via a worker pool, ingest chargeback labels via a feedback API for daily Platt micro-calibration, and derives temporal features server-side with a separate backtest endpoint.

## Problem Frame
The pipeline operates under a strict sub-10ms latency SLA. However, several codebase bottlenecks limit performance and reliability:
1. **Synchronous Caching Overhead:** Card history and features hashes are calculated and written within uvicorn-side task calls that query Redis lists and compute aggregates in Python.
2. **Sequential Queue Draining:** The prediction WAL queue is drained sequentially, causing queue bottlenecks under high transactional database load.
3. **Blocking Webhooks:** The alert worker subscribers await webhook deliveries sequentially; network lag on the receiver side blocks the entire alert dispatcher loop.
4. **Calibration Drift:** No dynamic feedback loop exists to update probability calibration coefficients based on real dispute labels.
5. **Payload Skew:** Client-calculated temporal features (`hour`) expose the pipeline to timezone skews and client tampering.

## Key Decisions
1. **Decoupled Write Queues:** Separate PostgreSQL WAL logging and Redis feature store updates into two independent queues (`queue:prediction_logs` and `queue:feature_updates`) to ensure database latency never slows down feature caching.
2. **Durable Webhook Queue:** Migrate webhook dispatches from raw Pub/Sub subscribers to a Redis list queue (`queue:alerts`) consumed by a pool of alert workers to guarantee delivery.
3. **API Feedback Ingestion:** Add a database migration to insert a nullable `true_label` column in the `predictions` table, and expose a `POST /feedback` endpoint to register dispute/chargeback outcomes.
4. **Zero-Input Inference Contract:** Remove the `hour` field from the transactional `POST /predict` API. Expose a separate `POST /predict/backtest` endpoint that accepts an optional `timestamp` parameter for historical runs.

## Requirements

### Serving and Interface
- **R1:** Remove the `hour` field from `PredictRequest` in `app/routes/predict.py`. Derive all temporal features (`Time`, `Time_Hours`, `Time_Normalized`, `Time_Hour`, `hour_sin`, `hour_cos`) directly from the server clock.
- **R2:** Create a `POST /predict/backtest` endpoint in `app/routes/predict.py` that accepts `card_id`, `amount`, and a custom `timestamp` override parameter for backtesting.
- **R3:** Create a `POST /feedback` endpoint in `app/routes/predict.py` that accepts a `prediction_id` and a boolean `true_label` (representing fraud dispute status) and updates the database row.

### Caching and Background Processing
- **R4:** Decouple Redis caching: the main uvicorn request path will push transaction updates to `queue:feature_updates` and offload updates to a background worker.
- **R5:** The feature cache worker must process `queue:feature_updates` immediately, writing card history and updating rolling feature aggregates in the Redis Hash.
- **R6:** Implement batch insertions in `workers/dlq_worker.py`: buffer enqueued logs from `queue:prediction_logs` and write them to PostgreSQL in bulk (using transaction batches of up to 100 items or 100ms timeout).
- **R7:** The alert worker in `workers/alert_worker.py` must consume alerts from a dedicated Redis queue `queue:alerts` using a task pool, ensuring webhook requests are non-blocking and durable.

### Machine Learning Calibration
- **R8:** Add an Alembic migration to add a nullable `true_label` column to the `predictions` database table.
- **R9:** Create a daily micro-calibration script in `model/src/calibrate_probabilities.py` that queries predictions with feedback labels, fits a post-hoc Platt scaling `LogisticRegression` calibration wrapper on model logits, and outputs the updated parameters.

## Scope Boundaries
- **In Scope:** Refactoring routes in `app/routes/predict.py`, background queues in `app/services/redis_cache.py`, updating worker logic in `workers/dlq_worker.py` and `workers/alert_worker.py`, database schema migrations, and Platt scaling updates.
- **Out of Scope:** Modifying payment gateway clearing parameters, full model uvicorn retraining routines, and Next.js frontend redesigns (outside of fixing compile warnings).

## Sources / Research
- Serving route: `app/routes/predict.py`
- Database worker: `workers/dlq_worker.py`
- Alert dispatch: `workers/alert_worker.py`
- Redis services: `app/services/redis_cache.py`
- Calibration logic: `model/src/calibrate_probabilities.py`
