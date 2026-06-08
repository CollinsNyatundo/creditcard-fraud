# Phase 3: Observability & Streaming — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Build the real-time observability layer: Redis Pub/Sub fan-out from the inference path to
WebSocket clients with replay semantics, plus a fraud analyst alert worker that fires outbound
webhooks for high-confidence detections.

**Architecture:** Background tasks publish to Redis channels. The `/stream` WebSocket endpoint
subscribes to Redis and broadcasts to connected clients. On reconnect, it replays recent rows from
PostgreSQL before switching to live push. The alert worker is an independent async process
subscribing to the `fraud:high_confidence_alerts` channel.

**Tech Stack:** FastAPI WebSockets · redis.asyncio Pub/Sub · asyncpg · httpx (webhook client) ·
pytest-asyncio

**Design decisions reference:** `docs/design_decisions.md` — S-4, U-1, U-4

**Pre-condition:** Phase 2 gate passed. `/predict` live and pushing to Redis WAL.

---

### Task 1: Add Redis Pub/Sub publisher to inference background task

**Files:**
- Create: `app/services/stream_publisher.py`
- Modify: `app/routes/predict.py` — call publish after WAL enqueue
- Create: `tests/unit/test_stream_publisher.py`

**Step 1: Write `app/services/stream_publisher.py`**

```python
"""Redis Pub/Sub publisher for live stream and alert channels. Resolution: S-4, U-1."""
import json
from datetime import datetime, timezone
import redis.asyncio as aioredis
from app.services.redis_cache import get_redis

STREAM_CHANNEL = "fraud:stream"
ALERT_CHANNEL = "fraud:high_confidence_alerts"


async def publish_transaction(
    prediction_id: str,
    card_id: str,
    amount: float,
    fraud_probability: float,
    is_flagged: bool,
    latency_ms: float,
) -> None:
    """Publish transaction event to the live stream channel."""
    redis: aioredis.Redis = await get_redis()
    payload = json.dumps({
        "prediction_id": prediction_id,
        "card_id": card_id,
        "amount": round(amount, 2),
        "fraud_probability": round(fraud_probability, 6),
        "is_flagged": is_flagged,
        "latency_ms": round(latency_ms, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    await redis.publish(STREAM_CHANNEL, payload)


async def publish_alert(
    prediction_id: str,
    card_id: str,
    amount: float,
    fraud_probability: float,
) -> None:
    """Publish high-confidence fraud alert to analyst alert channel."""
    redis: aioredis.Redis = await get_redis()
    payload = json.dumps({
        "prediction_id": prediction_id,
        "card_id": card_id,
        "amount": round(amount, 2),
        "probability": round(fraud_probability, 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    await redis.publish(ALERT_CHANNEL, payload)
```

**Step 2: Update `app/routes/predict.py` — add publish task**

In the background tasks block, after SHAP, add:

```python
    from app.services.stream_publisher import publish_transaction, publish_alert
    background_tasks.add_task(
        publish_transaction,
        prediction_id="",
        card_id=body.card_id,
        amount=body.amount,
        fraud_probability=prob,
        is_flagged=is_flagged,
        latency_ms=latency_ms,
    )
    if prob >= alert_threshold:
        background_tasks.add_task(
            publish_alert,
            prediction_id="",
            card_id=body.card_id,
            amount=body.amount,
            fraud_probability=prob,
        )
```

**Step 3: Write failing test**

```python
# tests/unit/test_stream_publisher.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_publish_transaction_calls_redis_publish():
    mock_redis = AsyncMock()
    with patch("app.services.stream_publisher.get_redis", return_value=mock_redis):
        from app.services.stream_publisher import publish_transaction, STREAM_CHANNEL
        await publish_transaction("pid", "card_1", 200.0, 0.87, True, 3.1)
    mock_redis.publish.assert_called_once()
    channel = mock_redis.publish.call_args[0][0]
    assert channel == STREAM_CHANNEL


@pytest.mark.asyncio
async def test_publish_alert_calls_alert_channel():
    mock_redis = AsyncMock()
    with patch("app.services.stream_publisher.get_redis", return_value=mock_redis):
        from app.services.stream_publisher import publish_alert, ALERT_CHANNEL
        await publish_alert("pid", "card_1", 200.0, 0.95)
    channel = mock_redis.publish.call_args[0][0]
    assert channel == ALERT_CHANNEL
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_stream_publisher.py -v
# Expected: 2 PASSED
```

**Step 5: Commit**

```bash
git add app/services/stream_publisher.py tests/unit/test_stream_publisher.py app/routes/predict.py
git commit -m "feat: add Redis Pub/Sub publisher for live stream and alert channels"
```

---

### Task 2: Build `/stream` WebSocket endpoint with replay semantics

**Files:**
- Create: `app/routes/stream.py`
- Modify: `app/main.py` — include stream router
- Create: `tests/unit/test_stream_endpoint.py`

**Step 1: Write `app/routes/stream.py`**

```python
"""WebSocket /stream endpoint with catch-up replay from PostgreSQL. Resolution: S-4, U-4."""
import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator
import redis.asyncio as aioredis
import sqlalchemy as sa
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.config import get_settings
from app.db.engine import AsyncSessionLocal
from app.services.redis_cache import get_redis

router = APIRouter()
settings = get_settings()

STREAM_CHANNEL = "fraud:stream"


async def _replay_from_db(
    websocket: WebSocket,
    since: datetime | None,
) -> None:
    """Fetch historical rows from PostgreSQL and send as replay burst."""
    if since is None:
        cutoff = datetime.now(timezone.utc).timestamp() - settings.stream_default_replay_seconds
        since = datetime.fromtimestamp(cutoff, tz=timezone.utc)

    max_cutoff = datetime.now(timezone.utc).timestamp() - settings.stream_max_replay_seconds
    max_since = datetime.fromtimestamp(max_cutoff, tz=timezone.utc)
    if since < max_since:
        since = max_since  # cap replay window

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa.text("""
                SELECT id::text, card_id, amount, fraud_probability,
                       is_flagged, latency_ms, created_at
                FROM predictions
                WHERE created_at >= :since
                ORDER BY created_at ASC
                LIMIT 500
            """),
            {"since": since},
        )
        rows = result.fetchall()

    for row in rows:
        await websocket.send_text(json.dumps({
            "type": "replay",
            "prediction_id": row[0],
            "card_id": row[1],
            "amount": float(row[2]),
            "fraud_probability": row[3],
            "is_flagged": row[4],
            "latency_ms": row[5],
            "created_at": row[6].isoformat(),
        }))


@router.websocket("/stream")
async def stream(
    websocket: WebSocket,
    since: str | None = Query(default=None),
):
    """WebSocket live fraud stream.

    Optional query param: ?since=<ISO-8601 timestamp> for replay.
    Replays up to 10 minutes of history, then switches to live.
    Drops messages older than STREAM_STALE_THRESHOLD_SECONDS.
    """
    await websocket.accept()

    # Step 1: Replay catch-up
    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            await websocket.send_text(json.dumps({"error": "invalid since format"}))
            await websocket.close(code=1003)
            return

    await _replay_from_db(websocket, since_dt)

    # Step 2: Switch to live Pub/Sub
    redis: aioredis.Redis = await get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(STREAM_CHANNEL)

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            payload = json.loads(message["data"])

            # Staleness check — drop messages older than threshold
            created_at = datetime.fromisoformat(payload["created_at"])
            age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age_seconds > settings.stream_stale_threshold_seconds:
                continue

            await websocket.send_text(json.dumps({**payload, "type": "live"}))
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(STREAM_CHANNEL)
        await pubsub.close()
```

**Step 2: Register router in `app/main.py`**

```python
from app.routes.stream import router as stream_router
app.include_router(stream_router)
```

**Step 3: Write failing test**

```python
# tests/unit/test_stream_endpoint.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def test_stream_rejects_bad_since_format():
    """WebSocket closes with error on invalid ISO timestamp."""
    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.db.engine.engine.dispose", new_callable=AsyncMock), \
         patch("app.services.redis_cache.get_redis", new_callable=AsyncMock), \
         patch("app.routes.stream.AsyncSessionLocal"):
        from app.main import app
        client = TestClient(app)
        with client.websocket_connect("/stream?since=not-a-date") as ws:
            msg = ws.receive_text()
            data = json.loads(msg)
            assert "error" in data
```

**Step 4: Run test**

```bash
pytest tests/unit/test_stream_endpoint.py -v
# Expected: PASSED
```

**Step 5: Commit**

```bash
git add app/routes/stream.py tests/unit/test_stream_endpoint.py app/main.py
git commit -m "feat: add /stream WebSocket with PostgreSQL replay and Redis Pub/Sub live push"
```

---

### Task 3: Implement alert worker

**Files:**
- Create: `workers/alert_worker.py`
- Create: `tests/unit/test_alert_worker.py`

**Step 1: Write `workers/alert_worker.py`**

```python
"""Fraud alert worker: subscribes to alert channel, fires outbound webhook. Resolution: U-1."""
import asyncio
import json
import logging
import httpx
import redis.asyncio as aioredis
from app.services.redis_cache import get_redis
from app.services.config_service import config_service
from app.services.stream_publisher import ALERT_CHANNEL

logger = logging.getLogger(__name__)


async def fire_webhook(url: str, payload: dict) -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Alert sent: %s → %d", payload.get("prediction_id"), response.status_code)
        except httpx.HTTPError as exc:
            logger.error("Webhook failed for %s: %s", payload.get("prediction_id"), exc)


async def alert_loop(redis: aioredis.Redis) -> None:
    pubsub = redis.pubsub()
    await pubsub.subscribe(ALERT_CHANNEL)
    logger.info("Alert worker subscribed to %s", ALERT_CHANNEL)

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue

        alert_enabled = await config_service.get("alert_enabled", default="false")
        if alert_enabled.lower() != "true":
            continue

        webhook_url = await config_service.get("alert_webhook_url", default="")
        if not webhook_url:
            logger.warning("Alert received but alert_webhook_url is not configured.")
            continue

        payload = json.loads(message["data"])
        await fire_webhook(webhook_url, payload)


async def main() -> None:
    redis = await get_redis()
    await alert_loop(redis)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Write failing test**

```python
# tests/unit/test_alert_worker.py
import pytest
import json
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_webhook_not_called_when_alert_disabled():
    from workers.alert_worker import alert_loop

    mock_redis = AsyncMock()
    mock_pubsub = AsyncMock()
    alert_payload = json.dumps({"prediction_id": "p1", "probability": 0.96})
    mock_pubsub.listen = AsyncMock(return_value=aiter([
        {"type": "message", "data": alert_payload},
    ]))
    mock_redis.pubsub.return_value = mock_pubsub

    with patch("workers.alert_worker.config_service.get",
               new_callable=AsyncMock, return_value="false"), \
         patch("workers.alert_worker.fire_webhook", new_callable=AsyncMock) as mock_fire:
        await alert_loop(mock_redis)

    mock_fire.assert_not_called()


def aiter(items):
    async def _gen():
        for item in items:
            yield item
    return _gen()


@pytest.mark.asyncio
async def test_webhook_called_when_alert_enabled():
    from workers.alert_worker import alert_loop

    mock_redis = AsyncMock()
    mock_pubsub = AsyncMock()
    payload = json.dumps({"prediction_id": "p2", "probability": 0.97})
    mock_pubsub.listen = AsyncMock(return_value=aiter([
        {"type": "message", "data": payload},
    ]))
    mock_redis.pubsub.return_value = mock_pubsub

    config_responses = iter(["true", "https://hooks.slack.com/test"])

    with patch("workers.alert_worker.config_service.get",
               side_effect=lambda *a, **kw: AsyncMock(return_value=next(config_responses))()),\
         patch("workers.alert_worker.fire_webhook", new_callable=AsyncMock) as mock_fire:
        await alert_loop(mock_redis)

    mock_fire.assert_called_once()
```

**Step 3: Run tests**

```bash
pytest tests/unit/test_alert_worker.py -v
# Expected: 2 PASSED
```

**Step 4: Commit**

```bash
git add workers/alert_worker.py tests/unit/test_alert_worker.py
git commit -m "feat: add alert worker subscribing to Redis channel and firing outbound webhooks"
```

---

### Phase 3 Done When

- [ ] All new tests pass: `pytest tests/unit/test_stream_publisher.py tests/unit/test_stream_endpoint.py tests/unit/test_alert_worker.py -v`
- [ ] Full suite still green: `pytest tests/ -v` (no regressions)
- [ ] Manual WebSocket test: `websocat ws://localhost:8000/stream?since=2026-06-08T00:00:00Z` returns replay rows then live events
- [ ] Alert worker logs "Alert sent" when `alert_enabled=true` and `fraud_probability > alert_threshold`
- [ ] All 3 tasks committed

> ✅ **Gate passed → proceed to [`2026-06-08-phase4-business-intelligence.md`](2026-06-08-phase4-business-intelligence.md)**
