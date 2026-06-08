"""Alert worker — subscribes to fraud:high_confidence_alerts and dispatches webhooks.

Design decision reference: U-1 (alert path for fraud analysts).

Full implementation: Phase 3 (observability layer).

This stub starts the event loop and logs a ready message so the
Docker container exits cleanly in Phase 1.

Phase 3 will replace this with:
    1. Redis SUBSCRIBE fraud:high_confidence_alerts
    2. Parse payload: {card_id, amount, probability, timestamp, prediction_id}
    3. Check system_config: alert_enabled, alert_threshold, alert_webhook_url
    4. Fire HTTP webhook (Slack / PagerDuty / email) via httpx
    5. Unit test: assert webhook NOT called when probability < alert_threshold
"""
import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Alert worker entry point (Phase 1 stub)."""
    logger.info("Alert worker started — awaiting Phase 3 implementation.")
    logger.info(
        "Alert enabled: %s | Webhook: %s",
        os.environ.get("ALERT_ENABLED", "false"),
        os.environ.get("ALERT_WEBHOOK_URL", "not configured"),
    )
    # Phase 1: keep alive so docker-compose health checks pass
    while True:
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
