# STRATEGY

## Target Problem
Fraud detection is too slow, opaque, and brittle for production card-not-present and real-time scoring workflows. Teams need a service that returns a trustworthy fraud signal fast enough for authorization decisions, with enough auditability to debug disputes and enough resilience to survive model or cache failures.

## Approach
Build a minimal, observable inference service around a single fraud model, using:
- rolling-window feature engineering from lightweight Redis state,
- manifest-validated model artifacts,
- async explanation and monitoring utilities,
- infrastructure that fails clearly and recovers locally when remote sources are unavailable.

## Persona
Primary user: backend/infra engineer integrating fraud scores into payment flows.
Pain points:
- latency spikes from history computation on every request,
- silent model/cache failures,
- missing traceability from score back to input history and model version.

## Key Metrics
1. P95 prediction latency
2. request success rate under Redis/cache failures
3. model manifest validation pass rate at startup
4. mean time to detect serving-side correctness issue from logs
5. rollout safety score for model/config changes

## Tracks
- Core inference path: `app/main.py`, `app/routes/predict.py`
- State and cache: `app/services/redis_cache.py`, `app/services/shap_service.py`
- Feature engineering: `src/`, `data/src/`
- Validation and observability: `utils/`, `tests/unit/`

## Non-Goals
- online model retraining or active learning
- multi-model orchestration beyond baseline + fallback cache
- replacing MLflow as the canonical model registry
- full payment processing or issuer-side dispute resolution

## Constraints
- single-MLflow model URI as source of truth during normal operations
- Redis used as hot feature cache only, not durable event store
- startup must fail closed on model manifest mismatch
- delivery surface is inference API, not batch data pipelines

## Current Learnings
1. **CE environment is now repeatable**: setup scripted dependencies (`ast-grep`, `fd`, `hyperfine`, `gh`, `jq`) and added `.compound-engineering/config.local.yaml`, which means future reviews/plans no longer depend on ad-hoc tool state.
2. **Redis feature history lowers per-request compute cost**: the current diff introduces a hash-based aggregation path and an extra GET path in `app/routes/predict.py`; it reduces recomputation but adds cache-miss branching that deserves measurement.
3. **Cache invalidation is the weakest link in the new flow**: TTL-only expiry plus 9-item windows creates potential staleness under burst traffic; clean invalidation likely matters more than extra aggregation formats.
4. **Request tracing is missing from the current refactor**: startup MLflow/local fallback and manifest validation events are logged, but per-request source/cache/model-version trace is not; this blocks fast debugging and violates metric #4 directly.
5. **Batch scoring remains an obvious gap**: real-time is the only path, which limits model evaluation, backtests, and retrospective alerting; this is a low-effort, high-value next surface.
6. **Tooling note**: `fd` is provided by `fdfind` via `~/.local/bin`; any scripts should call `fd`/`fdfind` consistently, with `fd` preferred.
