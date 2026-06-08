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
