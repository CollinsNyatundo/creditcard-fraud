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
