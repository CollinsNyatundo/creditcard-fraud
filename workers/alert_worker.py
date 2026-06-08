"""Fraud alert worker: subscribes to alert channel, fires outbound webhook with retries. Resolution: U-1."""
import asyncio
import json
import logging
import httpx
import redis.asyncio as aioredis
from app.services.redis_cache import get_redis
from app.services.config_service import config_service
from app.services.stream_publisher import ALERT_CHANNEL

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds


async def fire_webhook(url: str, payload: dict) -> None:
    """Send payload to webhook URL with exponential backoff retries."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                logger.info(
                    "Alert sent successfully (attempt %d): %s → %d",
                    attempt + 1,
                    payload.get("prediction_id"),
                    response.status_code,
                )
                return
            except (httpx.HTTPError, httpx.RequestError) as exc:
                logger.warning(
                    "Webhook dispatch attempt %d failed for prediction %s: %s",
                    attempt + 1,
                    payload.get("prediction_id"),
                    exc,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                else:
                    logger.error(
                        "Max webhook retries exceeded for prediction %s. Webhook URL: %s",
                        payload.get("prediction_id"),
                        url,
                    )


async def alert_loop(redis: aioredis.Redis) -> None:
    """Subscribe to the alert channel and process incoming notifications."""
    pubsub = redis.pubsub()
    await pubsub.subscribe(ALERT_CHANNEL)
    logger.info("Alert worker subscribed to channel: %s", ALERT_CHANNEL)

    try:
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

            try:
                payload = json.loads(message["data"])
                await fire_webhook(webhook_url, payload)
            except Exception as e:
                logger.error("Error processing alert message data: %s", e)
    except asyncio.CancelledError:
        logger.info("Alert loop cancelled.")
        raise
    except Exception as e:
        logger.error("Error in alert loop: %s", e)
    finally:
        await pubsub.unsubscribe(ALERT_CHANNEL)
        await pubsub.close()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    redis = await get_redis()
    logger.info("Starting Alert worker daemon...")
    await alert_loop(redis)


if __name__ == "__main__":
    asyncio.run(main())
