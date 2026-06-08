# Phase 2: Inference Path — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Build the authenticated `/predict` endpoint that completes in <4ms (model + Redis),
then offloads all logging, SHAP, and alerting to async background tasks using the WAL buffering
pattern to ensure zero silent data loss.

**Architecture:** Request path: `auth middleware → Redis fetch → feature engineering → LightGBM
→ return JSON`. Everything else (DB write, SHAP, alerts) is a `BackgroundTask` that writes first
to Redis WAL queue, then a drain worker writes to PostgreSQL with retries.

**Tech Stack:** FastAPI BackgroundTasks · asyncpg · Redis WAL queue · LightGBM · scikit-learn
RobustScaler · shap · Celery (DLQ worker) · pytest-asyncio

**Design decisions reference:** `docs/design_decisions.md` — S-1, S-2, S-3, C-4, U-1

**Pre-condition:** Phase 1 gate must be fully passed (`[x]` on all Phase 1 done-when items).

---

### Task 1: Create DB-backed runtime config service

**Files:**
- Create: `app/services/config_service.py`
- Create: `tests/unit/test_config_service.py`

**Step 1: Write `app/services/config_service.py`**

```python
"""Runtime config loaded from system_config table with in-memory TTL cache. Resolution: C-4."""
import asyncio
import time
from typing import Any
import sqlalchemy as sa
from app.db.engine import AsyncSessionLocal


class ConfigService:
    def __init__(self, ttl: int = 60) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.monotonic() < expires_at:
                    return value

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    sa.text("SELECT value FROM system_config WHERE key = :key"),
                    {"key": key},
                )
                row = result.fetchone()

            value = row[0] if row else default
            self._cache[key] = (value, time.monotonic() + self._ttl)
            return value

    async def get_float(self, key: str, default: float) -> float:
        raw = await self.get(key, str(default))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default


# Module-level singleton
config_service = ConfigService(ttl=60)
```

**Step 2: Write failing test**

```python
# tests/unit/test_config_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_get_returns_db_value_on_cache_miss():
    from app.services.config_service import ConfigService

    service = ConfigService(ttl=60)
    mock_row = MagicMock()
    mock_row.__getitem__ = MagicMock(return_value="0.75")

    with patch("app.services.config_service.AsyncSessionLocal") as mock_session:
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=mock_row))
        )
        result = await service.get_float("shap_trigger_threshold", default=0.5)

    assert result == 0.75


@pytest.mark.asyncio
async def test_get_returns_cached_value_on_cache_hit():
    from app.services.config_service import ConfigService
    import time

    service = ConfigService(ttl=60)
    service._cache["shap_trigger_threshold"] = ("0.80", time.monotonic() + 60)

    with patch("app.services.config_service.AsyncSessionLocal") as mock_session:
        result = await service.get_float("shap_trigger_threshold", default=0.5)
        mock_session.assert_not_called()

    assert result == 0.80
```

**Step 3: Run test**

```bash
pytest tests/unit/test_config_service.py -v
# Expected: 2 PASSED
```

**Step 4: Commit**

```bash
git add app/services/config_service.py tests/unit/test_config_service.py
git commit -m "feat: add DB-backed config service with 60s in-memory TTL cache"
```

---

### Task 2: Create WAL buffering writer service

**Files:**
- Create: `app/services/prediction_writer.py`
- Create: `tests/unit/test_prediction_writer.py`

**Step 1: Write `app/services/prediction_writer.py`**

```python
"""WAL buffering pattern for async DB writes. Resolution: S-1.

Write path: Redis WAL queue → drain worker → PostgreSQL.
This module handles the enqueue side. The drain worker is in workers/dlq_worker.py.
"""
import json
import uuid
from datetime import datetime, timezone
import redis.asyncio as aioredis
from app.config import get_settings
from app.services.redis_cache import get_redis

WAL_KEY = "queue:prediction_logs"
settings = get_settings()


async def enqueue_prediction_log(
    card_id: str,
    amount: float,
    fraud_probability: float,
    is_flagged: bool,
    threshold_used: float,
    latency_ms: float,
    prediction_id: str | None = None,
) -> str:
    """Push prediction log to Redis WAL queue. Returns prediction_id."""
    pid = prediction_id or str(uuid.uuid4())
    payload = {
        "id": pid,
        "card_id": card_id,
        "amount": amount,
        "fraud_probability": fraud_probability,
        "is_flagged": is_flagged,
        "threshold_used": threshold_used,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    redis: aioredis.Redis = await get_redis()
    await redis.rpush(WAL_KEY, json.dumps(payload))
    return pid
```

**Step 2: Write failing test**

```python
# tests/unit/test_prediction_writer.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_enqueue_pushes_to_wal_key():
    mock_redis = AsyncMock()
    with patch("app.services.prediction_writer.get_redis", return_value=mock_redis):
        from app.services.prediction_writer import enqueue_prediction_log, WAL_KEY
        pid = await enqueue_prediction_log(
            card_id="card_abc",
            amount=100.0,
            fraud_probability=0.91,
            is_flagged=True,
            threshold_used=0.5,
            latency_ms=3.2,
        )
    assert isinstance(pid, str) and len(pid) == 36  # UUID
    mock_redis.rpush.assert_called_once()
    call_args = mock_redis.rpush.call_args
    assert call_args[0][0] == WAL_KEY
```

**Step 3: Run test**

```bash
pytest tests/unit/test_prediction_writer.py -v
# Expected: PASSED
```

**Step 4: Commit**

```bash
git add app/services/prediction_writer.py tests/unit/test_prediction_writer.py
git commit -m "feat: add WAL prediction writer — enqueues to Redis before PostgreSQL write"
```

---

### Task 3: Build `/predict` endpoint

**Files:**
- Create: `app/routes/__init__.py`
- Create: `app/routes/predict.py`
- Modify: `app/main.py` — include router
- Create: `tests/unit/test_predict_endpoint.py`

**Step 1: Write `app/routes/predict.py`**

```python
"""POST /predict — inference path. Must complete < 4ms before background tasks fire."""
import time
import numpy as np
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.services.redis_cache import get_redis, push_card_amount, get_card_history
from app.services.prediction_writer import enqueue_prediction_log
from app.services.config_service import config_service

router = APIRouter()


class PredictRequest(BaseModel):
    card_id: str
    amount: float
    hour: int  # 0-23


class PredictResponse(BaseModel):
    prediction_id: str
    is_fraud: bool
    fraud_probability: float
    latency_ms: float


def _engineer_features(amount: float, hour: int, history: list[float]) -> np.ndarray:
    """In-memory feature engineering — mirrors training pipeline."""
    import math
    amount_log = math.log1p(amount)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    rolling_mean = float(np.mean(history)) if history else amount
    rolling_std = float(np.std(history)) if len(history) > 1 else 0.0
    # NOTE: V1-V28 are not available at runtime — set to 0 (batch mode only)
    v_features = [0.0] * 28
    return np.array([amount, *v_features, amount_log, hour_sin, hour_cos,
                     rolling_mean, rolling_std], dtype=np.float32).reshape(1, -1)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictResponse:
    t0 = time.perf_counter()
    model = request.app.state.model

    redis = await get_redis()
    history = await get_card_history(redis, body.card_id)

    features = _engineer_features(body.amount, body.hour, history)
    prob = float(model.predict(features)[0])

    threshold = await config_service.get_float("shap_trigger_threshold", default=0.5)
    is_flagged = prob >= threshold
    latency_ms = (time.perf_counter() - t0) * 1000

    # Fire-and-forget background tasks (after response is returned)
    background_tasks.add_task(push_card_amount, redis, body.card_id, body.amount)
    background_tasks.add_task(
        enqueue_prediction_log,
        card_id=body.card_id,
        amount=body.amount,
        fraud_probability=prob,
        is_flagged=is_flagged,
        threshold_used=threshold,
        latency_ms=latency_ms,
    )

    return PredictResponse(
        prediction_id="",  # filled by WAL writer
        is_fraud=is_flagged,
        fraud_probability=round(prob, 6),
        latency_ms=round(latency_ms, 3),
    )
```

**Step 2: Register router in `app/main.py`**

```python
from app.routes.predict import router as predict_router
app.include_router(predict_router)
```

**Step 3: Write failing test**

```python
# tests/unit/test_predict_endpoint.py
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([0.92])
    return m


@pytest.mark.asyncio
async def test_predict_returns_fraud_flag(mock_model):
    with patch("mlflow.lightgbm.load_model", return_value=mock_model), \
         patch("app.db.engine.engine.dispose", new_callable=AsyncMock), \
         patch("app.services.redis_cache.get_redis", new_callable=AsyncMock) as mock_redis, \
         patch("app.services.config_service.config_service.get_float",
               new_callable=AsyncMock, return_value=0.5), \
         patch("app.middleware.auth.AsyncSessionLocal") as mock_session:
        # Auth: valid key
        mock_redis.return_value.lrange = AsyncMock(return_value=[])
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=("some-id",)))
        )
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/predict",
                json={"card_id": "card_001", "amount": 500.0, "hour": 3},
                headers={"X-API-Key": "test-key"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == pytest.approx(0.92, abs=0.01)
    assert data["latency_ms"] < 100  # should be well under 100ms in test
```

**Step 4: Run test**

```bash
pytest tests/unit/test_predict_endpoint.py -v
# Expected: PASSED
```

**Step 5: Commit**

```bash
git add app/routes/ tests/unit/test_predict_endpoint.py app/main.py
git commit -m "feat: add /predict endpoint with <4ms inference path and async background offload"
```

---

### Task 4: Implement DLQ drain worker

**Files:**
- Create: `workers/__init__.py`
- Create: `workers/dlq_worker.py`
- Create: `tests/unit/test_dlq_worker.py`

**Step 1: Write `workers/dlq_worker.py`**

```python
"""DLQ drain worker: pops from Redis WAL, writes to PostgreSQL with retries. Resolution: S-1."""
import asyncio
import json
import logging
import sqlalchemy as sa
import redis.asyncio as aioredis
from app.services.redis_cache import get_redis, WAL_KEY  # re-export WAL_KEY
from app.db.engine import AsyncSessionLocal
from app.services.prediction_writer import WAL_KEY

logger = logging.getLogger(__name__)
DLQ_KEY = "queue:prediction_logs:dlq"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds


async def write_prediction(payload: dict) -> None:
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                sa.text("""
                    INSERT INTO predictions
                      (id, card_id, amount, fraud_probability, is_flagged,
                       threshold_used, latency_ms, created_at)
                    VALUES
                      (:id, :card_id, :amount, :fraud_probability, :is_flagged,
                       :threshold_used, :latency_ms, :created_at)
                    ON CONFLICT (id) DO NOTHING
                """),
                payload,
            )


async def drain_loop(redis: aioredis.Redis) -> None:
    while True:
        raw = await redis.blpop(WAL_KEY, timeout=5)
        if raw is None:
            continue
        _, item = raw
        payload = json.loads(item)
        for attempt in range(MAX_RETRIES):
            try:
                await write_prediction(payload)
                break
            except Exception as exc:
                logger.warning("DB write attempt %d failed: %s", attempt + 1, exc)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                else:
                    logger.error("Max retries exceeded — sending to DLQ: %s", payload["id"])
                    await redis.rpush(DLQ_KEY, item)


async def main() -> None:
    redis = await get_redis()
    logger.info("DLQ worker started — draining %s", WAL_KEY)
    await drain_loop(redis)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Write failing test**

```python
# tests/unit/test_dlq_worker.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_drain_writes_to_db_on_success():
    payload = {
        "id": "test-uuid",
        "card_id": "c1",
        "amount": 50.0,
        "fraud_probability": 0.3,
        "is_flagged": False,
        "threshold_used": 0.5,
        "latency_ms": 2.1,
        "created_at": "2026-06-08T10:00:00+00:00",
    }
    mock_redis = AsyncMock()
    # blpop returns once, then raises StopAsyncIteration to end loop
    mock_redis.blpop = AsyncMock(side_effect=[
        ("queue:prediction_logs", json.dumps(payload).encode()),
        asyncio.CancelledError,
    ])

    with patch("workers.dlq_worker.AsyncSessionLocal") as mock_session:
        mock_ctx = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.begin.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.execute = AsyncMock()

        import asyncio
        from workers.dlq_worker import drain_loop
        try:
            await drain_loop(mock_redis)
        except asyncio.CancelledError:
            pass

    mock_ctx.execute.assert_called_once()


@pytest.mark.asyncio
async def test_drain_pushes_to_dlq_after_max_retries():
    import asyncio
    import json
    from unittest.mock import AsyncMock, patch
    payload = {"id": "fail-uuid", "card_id": "c2", "amount": 0.0,
               "fraud_probability": 0.0, "is_flagged": False,
               "threshold_used": 0.5, "latency_ms": 1.0, "created_at": "2026-06-08T00:00:00+00:00"}
    mock_redis = AsyncMock()
    mock_redis.blpop = AsyncMock(side_effect=[
        ("queue:prediction_logs", json.dumps(payload).encode()),
        asyncio.CancelledError,
    ])

    with patch("workers.dlq_worker.AsyncSessionLocal") as mock_session, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_session.return_value.__aenter__.return_value.begin.return_value.__aenter__ \
            = AsyncMock(side_effect=Exception("DB down"))

        from workers.dlq_worker import drain_loop
        try:
            await drain_loop(mock_redis)
        except asyncio.CancelledError:
            pass

    mock_redis.rpush.assert_called_once()
```

**Step 3: Run tests**

```bash
pytest tests/unit/test_dlq_worker.py -v
# Expected: 2 PASSED
```

**Step 4: Commit**

```bash
git add workers/ tests/unit/test_dlq_worker.py
git commit -m "feat: add DLQ drain worker with exponential backoff and dead-letter queue fallback"
```

---

### Task 5: Add selective async SHAP background task

**Files:**
- Create: `app/services/shap_service.py`
- Create: `tests/unit/test_shap_service.py`
- Modify: `app/routes/predict.py` — add SHAP background task call

**Step 1: Write `app/services/shap_service.py`**

```python
"""Selective TreeSHAP computation for flagged transactions. Resolution: C-4, D-3."""
import numpy as np
import shap
import sqlalchemy as sa
from app.db.engine import AsyncSessionLocal
from app.services.config_service import config_service


async def compute_and_store_shap(
    prediction_id: str,
    model,
    features: np.ndarray,
    feature_names: list[str],
) -> None:
    """Calculate local SHAP values and write to shap_explanations table.
    Called only for transactions above the shap_trigger_threshold.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)[0]  # shape: (n_features,)

    # Fetch human-readable names
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa.text("SELECT feature_name, human_readable FROM feature_explanations")
        )
        name_map = {row[0]: row[1] for row in result.fetchall()}

    rows = [
        {
            "prediction_id": prediction_id,
            "feature_name": name,
            "shap_value": float(val),
            "human_readable": name_map.get(name),
        }
        for name, val in zip(feature_names, shap_values)
    ]

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                sa.text("""
                    INSERT INTO shap_explanations
                      (prediction_id, feature_name, shap_value, human_readable)
                    VALUES (:prediction_id, :feature_name, :shap_value, :human_readable)
                """),
                rows,
            )
```

**Step 2: Update `app/routes/predict.py` — add conditional SHAP task**

Add after `enqueue_prediction_log` background task:

```python
    alert_threshold = await config_service.get_float("alert_threshold", default=0.90)
    if is_flagged:
        from app.services.shap_service import compute_and_store_shap
        background_tasks.add_task(
            compute_and_store_shap,
            prediction_id="",  # WAL writer sets real ID
            model=model,
            features=features,
            feature_names=FEATURE_NAMES,  # define constant at top of file
        )
    if prob >= alert_threshold:
        from app.services.alert_service import enqueue_alert
        background_tasks.add_task(enqueue_alert, body.card_id, body.amount, prob)
```

**Step 3: Write failing test for SHAP service**

```python
# tests/unit/test_shap_service.py
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_shap_writes_rows_to_db():
    mock_model = MagicMock()
    features = np.zeros((1, 33))
    feature_names = [f"V{i}" for i in range(1, 29)] + \
                    ["Amount", "hour_sin", "hour_cos", "rolling_mean", "rolling_std"]

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [np.ones(33) * 0.1]

    with patch("shap.TreeExplainer", return_value=mock_explainer), \
         patch("app.services.shap_service.AsyncSessionLocal") as mock_session:
        mock_ctx = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
        mock_ctx.begin.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.services.shap_service import compute_and_store_shap
        await compute_and_store_shap("pred-123", mock_model, features, feature_names)

    # execute called: once for name map, once for insert
    assert mock_ctx.execute.call_count == 2
```

**Step 4: Run test**

```bash
pytest tests/unit/test_shap_service.py -v
# Expected: PASSED
```

**Step 5: Commit**

```bash
git add app/services/shap_service.py tests/unit/test_shap_service.py app/routes/predict.py
git commit -m "feat: add selective async SHAP computation for flagged transactions"
```

---

### Phase 2 Done When

- [ ] `pytest tests/unit/test_config_service.py tests/unit/test_prediction_writer.py tests/unit/test_predict_endpoint.py tests/unit/test_dlq_worker.py tests/unit/test_shap_service.py -v` → all **PASSED**
- [ ] `curl -X POST http://localhost:8000/predict -H "X-API-Key: <key>" -d '{"card_id":"c1","amount":500,"hour":2}'` returns 200 with `fraud_probability` and `latency_ms < 10`
- [ ] Redis WAL key `queue:prediction_logs` receives payloads (verify with `redis-cli llen queue:prediction_logs`)
- [ ] DLQ worker container drains queue and rows appear in `predictions` table
- [ ] All 5 tasks committed

> ✅ **Gate passed → proceed to [`2026-06-08-phase3-observability.md`](2026-06-08-phase3-observability.md)**
