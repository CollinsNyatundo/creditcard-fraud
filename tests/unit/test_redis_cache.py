"""Tests: Redis feature cache service — LTRIM hard-cap pattern.

Verifies:
- push_card_amount uses an atomic pipeline with rpush + ltrim + expire (S-2).
- ltrim is always called with -WINDOW_SIZE to enforce the hard cap.
- get_card_history returns floats parsed from string values.

Design decision reference: S-2 (Redis list unbounded growth prevention)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, call

from app.services.redis_cache import (
    WINDOW_SIZE,
    TTL_SECONDS,
    get_card_history,
    push_card_amount,
    push_to_prediction_log_queue,
)


def _make_mock_redis_with_pipeline() -> tuple[MagicMock, AsyncMock]:
    """Helper: returns (mock_redis, mock_pipe) with pipeline context manager wired up."""
    mock_pipe = AsyncMock()
    # rpush, ltrim, expire are regular sync calls on the pipe object
    mock_pipe.rpush = MagicMock()
    mock_pipe.ltrim = MagicMock()
    mock_pipe.expire = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[1, 1, True])

    mock_redis = MagicMock()
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)

    return mock_redis, mock_pipe


@pytest.mark.asyncio
async def test_push_uses_ltrim_pipeline() -> None:
    """push_card_amount must use rpush + ltrim + expire atomically (S-2)."""
    mock_redis, mock_pipe = _make_mock_redis_with_pipeline()

    await push_card_amount(mock_redis, "card_123", 42.50)

    mock_redis.pipeline.assert_called_once_with(transaction=True)
    mock_pipe.rpush.assert_called_once_with("card:card_123:history", "42.5")
    mock_pipe.ltrim.assert_called_once_with("card:card_123:history", -WINDOW_SIZE, -1)
    mock_pipe.expire.assert_called_once_with("card:card_123:history", TTL_SECONDS)
    mock_pipe.execute.assert_called_once()


@pytest.mark.asyncio
async def test_push_window_size_used_for_ltrim() -> None:
    """ltrim must use -WINDOW_SIZE, not a hardcoded value."""
    mock_redis, mock_pipe = _make_mock_redis_with_pipeline()

    await push_card_amount(mock_redis, "card_XYZ", 100.0)

    # The second arg to ltrim must equal -WINDOW_SIZE (currently -10)
    ltrim_call = mock_pipe.ltrim.call_args
    assert ltrim_call[0][1] == -WINDOW_SIZE, (
        f"Expected ltrim start={-WINDOW_SIZE}, got {ltrim_call[0][1]}"
    )


@pytest.mark.asyncio
async def test_push_correct_redis_key_format() -> None:
    """Redis key must be card:{card_id}:history."""
    mock_redis, mock_pipe = _make_mock_redis_with_pipeline()

    await push_card_amount(mock_redis, "my-card-id", 9.99)

    expected_key = "card:my-card-id:history"
    mock_pipe.rpush.assert_called_once_with(expected_key, "9.99")
    mock_pipe.ltrim.assert_called_once_with(expected_key, -WINDOW_SIZE, -1)
    mock_pipe.expire.assert_called_once_with(expected_key, TTL_SECONDS)


@pytest.mark.asyncio
async def test_get_card_history_returns_floats() -> None:
    """get_card_history must parse string values from Redis into floats."""
    mock_redis = AsyncMock()
    mock_redis.lrange = AsyncMock(return_value=["10.5", "20.0", "30.75"])

    result = await get_card_history(mock_redis, "card_456")

    assert result == [10.5, 20.0, 30.75]
    mock_redis.lrange.assert_called_once_with("card:card_456:history", 0, -1)


@pytest.mark.asyncio
async def test_get_card_history_empty() -> None:
    """get_card_history returns an empty list for a card with no history."""
    mock_redis = AsyncMock()
    mock_redis.lrange = AsyncMock(return_value=[])

    result = await get_card_history(mock_redis, "new_card")

    assert result == []


@pytest.mark.asyncio
async def test_push_to_prediction_log_queue() -> None:
    """push_to_prediction_log_queue pushes payload to queue:prediction_logs (S-1)."""
    mock_redis = AsyncMock()
    mock_redis.rpush = AsyncMock(return_value=1)

    await push_to_prediction_log_queue(mock_redis, '{"card_id": "abc"}')

    mock_redis.rpush.assert_called_once_with(
        "queue:prediction_logs", '{"card_id": "abc"}'
    )
