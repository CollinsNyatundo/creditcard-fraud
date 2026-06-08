# tests/unit/test_alert_worker.py
import pytest
import json
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from workers.alert_worker import fire_webhook, alert_loop


def aiter(items):
    async def _gen():
        for item in items:
            yield item
    return _gen()


@pytest.mark.asyncio
async def test_webhook_not_called_when_alert_disabled():
    mock_redis = AsyncMock()
    mock_pubsub = AsyncMock()
    alert_payload = json.dumps({"prediction_id": "p1", "probability": 0.96})
    mock_pubsub.listen = MagicMock(return_value=aiter([
        {"type": "message", "data": alert_payload},
    ]))
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

    with patch("workers.alert_worker.config_service.get",
               new_callable=AsyncMock, return_value="false"), \
         patch("workers.alert_worker.fire_webhook", new_callable=AsyncMock) as mock_fire:
        await alert_loop(mock_redis)

    mock_fire.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_called_when_alert_enabled():
    mock_redis = AsyncMock()
    mock_pubsub = AsyncMock()
    payload = json.dumps({"prediction_id": "p2", "probability": 0.97})
    mock_pubsub.listen = MagicMock(return_value=aiter([
        {"type": "message", "data": payload},
    ]))
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

    config_responses = iter(["true", "https://hooks.slack.com/test"])
    mock_get = AsyncMock(side_effect=lambda *a, **kw: next(config_responses))

    with patch("workers.alert_worker.config_service.get", new=mock_get),\
         patch("workers.alert_worker.fire_webhook", new_callable=AsyncMock) as mock_fire:
        await alert_loop(mock_redis)

    mock_fire.assert_called_once()


@pytest.mark.asyncio
async def test_fire_webhook_retries_and_succeeds():
    url = "http://fake-webhook.url"
    payload = {"prediction_id": "p3"}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    mock_post = AsyncMock(side_effect=[
        httpx.RequestError("Connection Refused"),
        mock_response
    ])

    with patch("httpx.AsyncClient.post", new=mock_post), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await fire_webhook(url, payload)

    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_fire_webhook_retries_exhausted():
    url = "http://fake-webhook.url"
    payload = {"prediction_id": "p4"}

    mock_post = AsyncMock(side_effect=httpx.RequestError("Database connection lost"))

    with patch("httpx.AsyncClient.post", new=mock_post), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await fire_webhook(url, payload)

    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2
