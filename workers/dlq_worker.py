"""DLQ drain worker: pops from Redis WAL, writes to PostgreSQL with retries. Resolution: S-1."""
import asyncio
import json
import logging
import sqlalchemy as sa
import redis.asyncio as aioredis
from app.db.engine import AsyncSessionLocal
from app.services.redis_cache import get_redis
from app.services.prediction_writer import WAL_KEY

logger = logging.getLogger(__name__)
DLQ_KEY = "queue:prediction_logs:dlq"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds


async def write_prediction(payload: dict) -> None:
    """Insert prediction payload into predictions table."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Parse ISO datetime string to python datetime object for SQLAlchemy/asyncpg
            from datetime import datetime
            created_at_str = payload.get("created_at")
            if created_at_str:
                if created_at_str.endswith("Z"):
                    created_at_str = created_at_str[:-1] + "+00:00"
                payload["created_at"] = datetime.fromisoformat(created_at_str)
            else:
                payload["created_at"] = datetime.now()

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
    """Infinite loop draining prediction logs from Redis to PG."""
    while True:
        try:
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
                        logger.error("Max retries exceeded — sending to DLQ: %s", payload.get("id"))
                        await redis.rpush(DLQ_KEY, item)
        except asyncio.CancelledError:
            logger.info("Drain loop cancelled.")
            raise
        except Exception as e:
            logger.error("Error in drain loop: %s", e)
            await asyncio.sleep(1.0)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    redis = await get_redis()
    logger.info("DLQ worker started — draining %s", WAL_KEY)
    await drain_loop(redis)


if __name__ == "__main__":
    asyncio.run(main())
