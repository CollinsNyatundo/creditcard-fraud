"""Dead Letter Queue worker — drains queue:prediction_logs into PostgreSQL.

Design decision reference: S-1 (WAL buffering pattern for async DB writes).

Full implementation: Phase 2 (inference path).

This stub starts the event loop and logs a ready message so the
Docker container exits cleanly in Phase 1 without crashing the
docker-compose stack.

Phase 2 will replace this with:
    1. Infinite loop polling redis.blpop("queue:prediction_logs")
    2. Exponential backoff PostgreSQL writes (max 3 retries)
    3. On all retries exhausted: re-enqueue to DLQ Celery task
    4. prometheus_client counter: prediction_log_failures_total
"""
import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    """DLQ worker entry point (Phase 1 stub)."""
    logger.info("DLQ worker started — awaiting Phase 2 implementation.")
    logger.info(
        "Redis URL: %s",
        os.environ.get("REDIS_URL", "not configured"),
    )
    # Phase 1: keep alive so docker-compose health checks pass
    while True:
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
