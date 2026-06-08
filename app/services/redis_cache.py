"""Redis feature cache service with atomic RPUSH + LTRIM hard-cap.

Design decision references:
- D-5: Redis sliding-window feature cache prevents DB overhead on every
  rolling feature calculation, keeping feature engineering under 1ms.
- S-2: Redis list unbounded growth is prevented by the atomic pipeline
  that hard-caps each card's history to WINDOW_SIZE items.

The LTRIM pattern:
    RPUSH card:{id}:history  <amount>
    LTRIM card:{id}:history  -10 -1   ← keep only last 10
    EXPIRE card:{id}:history 86400    ← 24-hour TTL

All three operations execute atomically in a single Redis pipeline
(MULTI/EXEC block), so there is no window where the list exceeds
WINDOW_SIZE even under concurrent writes.
"""
import redis.asyncio as aioredis

from app.config import get_settings

settings = get_settings()

WINDOW_SIZE: int = 10       # hard cap on list length (S-2)
TTL_SECONDS: int = 86_400   # 24-hour expiry


async def get_redis() -> aioredis.Redis:
    """Create and return an async Redis client from settings."""
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def push_card_amount(
    redis: aioredis.Redis,
    card_id: str,
    amount: float,
) -> None:
    """Atomically push a transaction amount and hard-cap the list.

    Guarantees: len(card:{card_id}:history) <= WINDOW_SIZE at all times,
    regardless of transaction volume. Safe under concurrent callers.

    Args:
        redis: An async Redis client instance.
        card_id: Unique card identifier (used as part of the Redis key).
        amount: Transaction amount to append to the card's history.
    """
    key: str = f"card:{card_id}:history"
    async with redis.pipeline(transaction=True) as pipe:
        pipe.rpush(key, str(amount))
        pipe.ltrim(key, -WINDOW_SIZE, -1)
        pipe.expire(key, TTL_SECONDS)
        await pipe.execute()


async def get_card_history(
    redis: aioredis.Redis,
    card_id: str,
) -> list[float]:
    """Return up to WINDOW_SIZE recent transaction amounts for a card.

    Args:
        redis: An async Redis client instance.
        card_id: Unique card identifier.

    Returns:
        List of floats (oldest → newest), length 0–WINDOW_SIZE.
    """
    key: str = f"card:{card_id}:history"
    raw: list[str] = await redis.lrange(key, 0, -1)
    return [float(x) for x in raw]


async def push_to_prediction_log_queue(
    redis: aioredis.Redis,
    payload: str,
) -> None:
    """Push a serialised prediction payload onto the WAL queue.

    This implements the Write-Ahead Log buffering pattern (S-1):
    the background task writes here first; the DLQ worker drains
    the queue and writes to PostgreSQL with retry logic.

    Args:
        redis: An async Redis client instance.
        payload: JSON-serialised prediction payload.
    """
    await redis.rpush("queue:prediction_logs", payload)
