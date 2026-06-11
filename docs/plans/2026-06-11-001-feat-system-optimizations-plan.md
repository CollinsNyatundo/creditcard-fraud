---
title: "feat: Decoupled System Optimizations and Hardening"
type: feat
status: active
date: 2026-06-11
origin: docs/brainstorms/2026-06-11-system-optimizations-requirements.md
---

# feat: Decoupled System Optimizations and Hardening

## Summary
Technical plan to implement a high-throughput, decoupled architecture for the Credit Card Fraud Detection Pipeline. It offloads database writes and feature caching to independent async queues, processes webhook alerts via a worker task pool, introduces API feedback loops for daily Platt micro-calibration, and simplifies transactional scoring to be strictly server-derived.

## Problem Frame
The current transactional path suffers from database write bottlenecks, blocking webhook notifications, and payload skews that limit serving throughput and reliability under a strict sub-10ms SLA. Moving calculations and database writes out of uvicorn-side task contexts into dedicated Redis-backed queues and batch workers resolves uvicorn process bottlenecks and ensures pipeline durability.

## Key Technical Decisions
1. **Queue Decoupling:** Introduce a new list key `queue:feature_updates` in Redis. Prediction requests will push transaction details here to be consumed by a dedicated Feature Cache Worker, removing uvicorn-side Redis history reads/writes from uvicorn-side task contexts.
2. **PostgreSQL Batch Logging:** Modify `workers/dlq_worker.py` to buffer popped logs in memory up to 100 entries or 100ms, then execute bulk inserts to PostgreSQL using a single transaction to eliminate write bottlenecks.
3. **Alert Webhook Queue & Worker Pool:** Replace Pub/Sub webhook calls in `workers/alert_worker.py` with a durable Redis queue (`queue:alerts`). The worker will manage a pool of concurrent tasks consuming this queue, preventing timeout delays on external webhooks from blocking the alert stream.
4. **Dispute Ingestion Schema:** Add a nullable boolean column `true_label` to the `predictions` table. Expose a `POST /feedback` endpoint to update this field on transaction rows using the original prediction ID.
5. **Zero-Input Temporal Inference:** Remove `hour` from the `POST /predict` payload. Calculate temporal features from the server clock. Create a separate `POST /predict/backtest` endpoint that accepts an optional `timestamp` to preserve historical scoring.

---

## Requirements

### Serving and Interface
- R1. Expose `POST /feedback` endpoint accepting `prediction_id` and a boolean `true_label`.
- R2. Expose `POST /predict/backtest` endpoint accepting `card_id`, `amount`, and `timestamp` (as float or ISO string).
- R3. Modify `POST /predict` to accept only `card_id` and `amount`. Derive temporal features using `time.time()` server-side.

### Caching and Background Workers
- R4. Replace uvicorn-side list reads/writes in uvicorn-side tasks with enqueuing events to `queue:feature_updates`.
- R5. Create `workers/feature_worker.py` to consume `queue:feature_updates` and update the Redis sliding window card history lists and pre-computed features hashes.
- R6. Update `workers/dlq_worker.py` to pool enqueued records from `queue:prediction_logs` in memory and write them to PostgreSQL in batch inserts.
- R7. Update `workers/alert_worker.py` to consume webhook triggers from `queue:alerts` using a task pool with limited concurrency.

### Calibration Loop
- R8. Implement daily micro-calibration fitting a Platt scaling `LogisticRegression` on LightGBM logits using feedback data where `true_label` is populated.

---

## Implementation Units

### U1. Database Migration for Dispute Feedback
- **Goal:** Add a nullable `true_label` column to the `predictions` database table.
- **Files:**
  - `alembic/versions/003_add_true_label_to_predictions.py`
- **Patterns:** Follow standard Alembic revision structure from `alembic/versions/001_canonical_schema.py`.
- **Test Scenarios:**
  - Verify `alembic upgrade head` adds `true_label` to the PostgreSQL schema.
  - Verify `alembic downgrade 002` removes it cleanly.

### U2. Serving Layer Payload Update & Zero-Input temporal features
- **Goal:** Update the transactional inference endpoint to remove client-calculated temporal parameters.
- **Files:**
  - `app/routes/predict.py`
- **Patterns:** Standard SlowAPI rate limiter and FastAPI Router parameters.
- **Test Scenarios:**
  - Verify `POST /predict` rejects payloads containing `hour`.
  - Verify `POST /predict` accepts payloads with only `card_id` and `amount` and succeeds.
  - Assert derived features match current server clock time.

### U3. Expose Predict Backtest and Feedback API endpoints
- **Goal:** Expose `/predict/backtest` and `/feedback` endpoints in the FastAPI router.
- **Files:**
  - `app/routes/predict.py`
- **Patterns:** Reuse `prepare_prediction_features` helper with custom timestamps.
- **Test Scenarios:**
  - Verify `/predict/backtest` scores transaction using custom input `timestamp`.
  - Verify `/feedback` updates the `true_label` column in PostgreSQL for the matching `prediction_id`.
  - Verify `/feedback` returns 404 for invalid/non-existent prediction IDs.

### U4. Decouple Redis Caching & Queue updates
- **Goal:** Update `app/services/redis_cache.py` to offload history updates to `queue:feature_updates`.
- **Files:**
  - `app/services/redis_cache.py`
  - `app/routes/predict.py`
- **Patterns:** Redis list push enqueuing (`rpush`).
- **Test Scenarios:**
  - Verify `/predict` pushes a JSON update payload to `queue:feature_updates` without calling uvicorn-side caching updates.

### U5. Feature Cache Worker daemon
- **Goal:** Create a background worker that consumes `queue:feature_updates` and maintains Redis feature hashes.
- **Files:**
  - `workers/feature_worker.py`
  - `docker-compose.yml`
- **Patterns:** Follow daemon framework of `workers/dlq_worker.py`.
- **Test Scenarios:**
  - Run worker, enqueue updates on `queue:feature_updates`, and assert Redis history list and features hash are correctly populated.

### U6. Batch DB writes in DLQ/WAL Worker
- **Goal:** Refactor `workers/dlq_worker.py` to buffer and batch log writes to PostgreSQL.
- **Files:**
  - `workers/dlq_worker.py`
- **Patterns:** Buffer queue pops in memory, bulk insert using SQLAlchemy core inserts.
- **Test Scenarios:**
  - Populate `queue:prediction_logs` with 150 items. Verify they are written in exactly 2 DB batches.
  - Verify the timeout trigger writes fewer than 100 items after 100ms.

### U7. Webhook worker with task pool and alerts queue
- **Goal:** Refactor `workers/alert_worker.py` to consume from `queue:alerts` using a limited task pool.
- **Files:**
  - `workers/alert_worker.py`
  - `app/routes/predict.py`
- **Patterns:** Task pool backed by a Semaphore to cap concurrent connections.
- **Test Scenarios:**
  - Enqueue multiple webhooks; verify they are dispatched concurrently without blocking the main event loop.
  - Assert failures on one webhook target do not block or delay delivery to other targets.

### U8. Daily Micro-Calibration Platt Scaling Task
- **Goal:** Implement the daily calibration job and ensure uvicorn lifespan reloading.
- **Files:**
  - `model/src/calibrate_probabilities.py`
  - `app/main.py`
- **Patterns:** Platt scaling wrapper training and reloading from cached pickle path.
- **Test Scenarios:**
  - Verify calibration logic trains a `LogisticRegression` on predictions with non-null `true_label`.
  - Assert that uvicorn lifespan reloads the calibrated model parameters at startup.

---

## Scope Boundaries
- **In Scope:** Decoupled Redis and PostgreSQL queues, batch log updates, non-blocking webhook alert tasks, feedback API endpoints, and Platt scaling scripts.
- **Out of Scope:** MLflow server migrations, model retraining scripts, and client-side web UI pages redesigns (other than code fixes).

## Risks & Dependencies
- **Feature Store Lag:** Decoupling updates to the feature cache worker means successive predict calls within milliseconds for the same card might score against stale transaction counts. (Mitigation: High-priority queue processing in uvicorn).
- **Alembic Table Locking:** Running database migrations on the `predictions` table under high write load. (Mitigation: Run during low-traffic maintenance window).
