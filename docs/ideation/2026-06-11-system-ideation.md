---
date: 2026-06-11
topic: Real-Time Credit Card Fraud Detection Pipeline Optimizations
focus: System Ideation and Structural Hardening
mode: repo-grounded
---

# Grounding Context

This ideation report is grounded in the architecture and runtime implementation of the Credit Card Fraud Detection Pipeline. The system is composed of five key architectural components:
1. **LightGBM Classifier & Calibrator**: A gradient-boosted tree model trained for binary fraud classification, paired with a Platt scaling calibration layer (wrapped in `IsotonicCalibratedBooster` which fits a `LogisticRegression` model on predictions) to output calibrated probabilities.
2. **FastAPI Serving Layer**: The high-throughput HTTP serving path that exposes prediction, token authorization, streaming, and health check endpoints.
3. **Redis Feature Store & WAL Cache**: Used for sliding-window transaction history lists, pre-computed feature caches, and a write-ahead log queue for predictions and alerts.
4. **PostgreSQL Database**: The long-term system of record for historical predictions, system configuration keys, and user credentials.
5. **Next.js Dashboard**: The operator dashboard that streams real-time transaction graphs, showcases metric summaries, and renders local explainability waterfalls.

### Key System Parameters & Thresholds
The following parameters dictate the pipeline's runtime behavior:
- **Decision Threshold**: `0.50` (read dynamically from PostgreSQL table `system_config` under the key `"shap_trigger_threshold"` in `app/routes/predict.py`). Transactions with calibrated probabilities greater than or equal to this threshold are flagged as fraud.
- **Alert Gate Threshold**: `0.90` (read dynamically from PostgreSQL under the key `"alert_threshold"` in `workers/alert_worker.py`). Outbound alerts are triggered only if the fraud probability meets this limit.
- **Latency Target**: `<10ms` SLA for predict requests (offloading heavy logging, feature cache updates, and explanation tasks to background workers).
- **Rate Limiting**: `100/second` on the `/predict` endpoint, enforced via SlowAPI middleware.
- **Window Size**: `10` transactions (sliding window cap for card histories in Redis).
- **Redis Cache TTL**: `86,400 seconds` (24 hours) for transaction history lists.
- **Background Task Retries**: Up to `3` attempts with exponential backoff for queue operations and webhook delivery.

### Scanned Codebase Friction Points
An audit of the codebase has identified several critical friction points and optimization gaps:
- **Sequential blocking in DLQ/WAL Worker (`workers/dlq_worker.py` lines 46-55)**: The queue drainage loop pops transaction logs from Redis (`blpop`) and awaits single PostgreSQL writes (`write_prediction`) sequentially. Under load, database latency or sleep-backed retries (exponential backoff) will back up the Redis queue, risking container memory exhaustion.
- **Sequential webhook execution in Alert Worker (`workers/alert_worker.py` lines 72-76)**: The worker listens to Redis Pub/Sub alerts and awaits `fire_webhook` sequentially. Any timeout, network lag, or backoff retry in downstream webhooks blocks the entire event loop, choking subsequent critical fraud alerts.
- **Missing imports in Replay View (`frontend/app/replay/page.tsx` lines 52, 88-95)**: The UI references `<Card />`, `<CardHeader />`, `<CardTitle />`, `<CardDescription />`, `<CardContent />`, and `<Clock />` layout components without importing them, causing static dashboard build compilation failures.
- **Backend API implementation gaps (`frontend/lib/api.ts` lines 59-75)**: The client attempts to query endpoints `/metrics` (for system status panels), `/config` (for dynamic parameters), and `/shap/{id}` (for the waterfall component). None of these endpoints are registered in the FastAPI app, causing persistent console 404 errors.
- **Confusing threshold parameter key naming (`app/routes/predict.py` line 79)**: The main transactional classification threshold is retrieved under the name `"shap_trigger_threshold"`. This misleading name implies it is only a trigger for explainability computations, creating technical debt and maintenance risks.

---

## Topic Axes

The optimization and hardening strategies are categorized along four orthogonal axes derived from the grounded scan:

### Axis 1: Latency, Throughput & SLA Enforcement
Focuses on the inference path to ensure that transaction-scoring latency remains under the 10ms SLA target. Optimizations in this axis target memory footprint, serialization overhead, and the CPU cost of real-time feature extraction.

### Axis 2: Algorithmic Calibration, Generalization & Feedback Loops
Covers the machine learning pipeline, including how prediction probabilities are calibrated against true drift, how classification boundaries are dynamically managed, and how downstream chargeback/feedback labels can be used to tune performance.

### Axis 3: Observability, Explainability & Auditing
Addresses telemetry, local explainability (SHAP), and model validation. Gaps in API metrics, performance monitoring, and explanation-loop starvation are mitigated here.

### Axis 4: Streaming, Background Processing & UI/UX Consistency
Targets background tasks and client-server alignment. This axis ensures workers handle high throughput without bottlenecks, and that client dashboards compile and display telemetry correctly.

---

## Ranked Ideas

### 1. Write-Time Feature Materialization with Incremental State Updates
- **Title**: Write-Time Feature Materialization with Incremental State Updates (O(1) read)
- **Description**: Currently, the serving path fetches the transaction history list and computes aggregates (`mean`, `std`, etc.) dynamically on every prediction request. This proposal shifts the computation to the write path. Each time a transaction is recorded, a Redis pipeline transaction (`MULTI/EXEC`) increments the aggregate values (`sum`, `sum_sq`, `count`, etc.) in a Redis Hash (`card:{card_id}:features`) using `HINCRBYFLOAT`. The read path then performs a single, highly efficient `HMGET` call to fetch pre-computed features.
- **Axis**: Axis 1: Latency, Throughput & SLA Enforcement
- **Basis**: `app/routes/predict.py` lines 140–196 (dynamic feature computation loop) and `app/services/redis_cache.py` lines 106–134 (Redis transaction history caching).
- **Rationale**: Reduces prediction feature computation time from $O(W)$ (where $W$ is the sliding window size) to $O(1)$ constant time. This reduces prediction hot path CPU utilization and ensures that latency remains well below the 10ms SLA even if the sliding window cap is expanded.
- **Downsides**: Increases write-path complexity and Redis memory utilization since aggregate fields are stored explicitly for each active card. Requires a migration script or a fallback mechanism during transition.
- **Confidence**: 95%
- **Complexity**: Medium
- **Status**: Unexplored

### 2. Asynchronous Batch Processing in WAL/DLQ Worker
- **Title**: Asynchronous Batch Processing in WAL/DLQ Worker
- **Description**: Replace the sequential blocking pop-and-write loop in `workers/dlq_worker.py` with a batch drainage pipeline. Instead of writing single transaction logs to PostgreSQL immediately, the worker will buffer popped logs in memory. It will trigger a bulk database write (using `INSERT INTO predictions VALUES (...) ON CONFLICT DO NOTHING`) either when the buffer reaches a threshold (e.g., 100 entries) or when a time limit (e.g., 100ms) is reached.
- **Axis**: Axis 4: Streaming, Background Processing & UI/UX Consistency
- **Basis**: `workers/dlq_worker.py` lines 46-55 (blocking loop popping predictions and writing sequentially).
- **Rationale**: Eliminates the database connection bottleneck. In the event of high concurrent transaction surges, writing logs in bulk transactions drastically reduces the number of roundtrips to PostgreSQL, preventing Redis WAL backlogs and memory starvation.
- **Downsides**: Introduces a minor delay (up to 100ms) before prediction logs are persisted to PostgreSQL. If the worker crashes, buffered logs in memory must be recovered (though they remain in the Redis list if using reliable queue popping patterns like `RPOPLPUSH`).
- **Confidence**: 90%
- **Complexity**: Medium
- **Status**: Unexplored

### 3. Event-Driven Non-Blocking Webhook Alerts
- **Title**: Event-Driven Non-Blocking Webhook Alerts
- **Description**: Refactor the alert dispatcher loop in `workers/alert_worker.py` to offload webhook HTTP calls to non-awaited background tasks. Rather than using `await fire_webhook(...)` directly in the Pub/Sub subscriber stream, the worker will use `asyncio.create_task(fire_webhook(...))` to dispatch the request. This allows the worker to continue polling the Pub/Sub stream immediately without waiting for HTTP handshakes, timeouts, or retries.
- **Axis**: Axis 4: Streaming, Background Processing & UI/UX Consistency
- **Basis**: `workers/alert_worker.py` lines 72-76 (awaiting webhook delivery sequentially blocks the Pub/Sub listener).
- **Rationale**: Guttering downstream webhook latency from the alert loop guarantees that a slow or unresponsive alert receiver does not block the delivery of alerts to other endpoints, solving queue choke points during high-velocity fraud incidents.
- **Downsides**: Requires limiting the maximum number of concurrent task threads (e.g., using `asyncio.Semaphore`) to avoid exhaustively opening outbound sockets or overloading client webhook endpoints.
- **Confidence**: 95%
- **Complexity**: Low
- **Status**: Unexplored

### 4. Daily Micro-Calibration Platt Scaling Layer
- **Title**: Daily Micro-Calibration Platt Scaling Layer
- **Description**: Implement a lightweight daily micro-calibration task that executes a Platt scaling fit (logistic regression on model logits) using confirmed labels from chargebacks and user disputes. The calibration parameters (scale and bias) are updated in the database and loaded by the model service at runtime, adapting the model's outputs to concept drift without full retraining.
- **Axis**: Axis 2: Algorithmic Calibration, Generalization & Feedback Loops
- **Basis**: `model/src/calibrate_probabilities.py` lines 35-75 (calibration logic) and `app/main.py` lines 110-116 (lifespan model loading).
- **Rationale**: Real-time fraud patterns shift quickly. Full model retraining is computationally heavy and requires deep validation. Daily micro-calibration of probability outputs corrects for model calibration drift and keeps the false positive rate stable with minimal overhead.
- **Downsides**: Relies on a steady stream of feedback labels. If incorrect or noisy labels are fed to the calibrator, it could corrupt probability score alignment.
- **Confidence**: 85%
- **Complexity**: Medium
- **Status**: Unexplored

### 5. Server-Side Zero-Input Time/Temporal Feature Derivation
- **Title**: Server-Side Zero-Input Time/Temporal Feature Derivation
- **Description**: Refactor the prediction endpoint `/predict` contract and model preprocessing to derive temporal features (`hour`, `day_of_week`, `is_weekend`) directly from the server system clock at the exact time the request is received. The client request payload is simplified to exclude client-calculated temporal parameters, accepting only raw inputs like `card_id`, `amount`, and `merchant_id`.
- **Axis**: Axis 1: Latency, Throughput & SLA Enforcement
- **Basis**: `app/routes/predict.py` line 491 (loading classification threshold) and `app/routes/predict.py` endpoint parameters.
- **Rationale**: Eliminates client-side clock skew, timezone translation issues, and potential client-side tampering of transaction timestamps. This encapsulates the feature engineering process within the serving boundary and simplifies the API integration contract for merchants.
- **Downsides**: Restricts the API's ability to easily backtest historical transactions out-of-the-box unless an optional override timestamp parameter is exposed in the request payload.
- **Confidence**: 95%
- **Complexity**: Low
- **Status**: Unexplored

---

## Rejection Summary

The following ideas were evaluated and rejected during the adversarial filtering phase:

### 1. Double-Consensus Card Blocking (Real-time Blocking)
- **Rationale for Rejection**: Integrating payment processor clearing endpoints to block transactions in real-time is out of scope. This is a payment clearance problem, not an ML pipeline optimization. It introduces complex compliance issues (e.g., PCI-DSS, 3D Secure, network rules) and massive third-party integration overhead that does not address the core performance or stability of the model scoring path itself.

### 2. Local In-Memory Card History Cache
- **Rationale for Rejection**: Storing sliding-window transaction details inside the FastAPI application process memory (e.g., in a local Python dictionary) rather than Redis. While this would eliminate network roundtrips to Redis, it fails in a production deployment where FastAPI is scaled across multiple uvicorn worker processes or containers. Workers would have inconsistent cached card states, violating transaction order and leading to corrupt feature calculations. It also risks application memory leaks under high concurrent transaction volume.

### 3. PostgreSQL Index Tuning for Prediction Logs
- **Rationale for Rejection**: Adding indexes to the PostgreSQL `predictions` table to optimize write latency is too tactical. PostgreSQL inserts are not throttled by indexes here; rather, the sequential blocking loop in the worker is the architectural bottleneck. Even with hyper-optimized indexes, awaiting single insertions sequentially will fail under high load. Optimizations must address the worker's processing logic (i.e., batching) rather than general database tuning.

### 4. Custom Rate Limiter Middleware
- **Rationale for Rejection**: Replacing SlowAPI with a custom-built token bucket middleware inside FastAPI. SlowAPI is already configured and handles client limits perfectly. Designing a custom rate limiter adds high refactoring risk and testing overhead with zero practical benefit to the system's prediction latency or throughput, representing premature optimization and NIH (Not Invented Here) syndrome.
