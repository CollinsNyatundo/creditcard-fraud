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
