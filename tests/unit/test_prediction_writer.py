# tests/unit/test_prediction_writer.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_enqueue_pushes_to_wal_key():
    mock_redis = AsyncMock()
    with patch("app.services.prediction_writer.get_redis", return_value=mock_redis):
        from app.services.prediction_writer import enqueue_prediction_log, WAL_KEY
        pid = await enqueue_prediction_log(
            card_id="card_abc",
            amount=100.0,
            fraud_probability=0.91,
            is_flagged=True,
            threshold_used=0.5,
            latency_ms=3.2,
        )
    assert isinstance(pid, str) and len(pid) == 36  # UUID
    mock_redis.rpush.assert_called_once()
    call_args = mock_redis.rpush.call_args
    assert call_args[0][0] == WAL_KEY
