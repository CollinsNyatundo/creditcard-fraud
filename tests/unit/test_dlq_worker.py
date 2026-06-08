# tests/unit/test_dlq_worker.py
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_drain_writes_to_db_on_success():
    payload = {
        "id": "test-uuid",
        "card_id": "c1",
        "amount": 50.0,
        "fraud_probability": 0.3,
        "is_flagged": False,
        "threshold_used": 0.5,
        "latency_ms": 2.1,
        "created_at": "2026-06-08T10:00:00+00:00",
    }
    mock_redis = AsyncMock()
    # blpop returns once, then raises CancelledError to end loop
    mock_redis.blpop = AsyncMock(side_effect=[
        ("queue:prediction_logs", json.dumps(payload).encode()),
        asyncio.CancelledError,
    ])

    with patch("workers.dlq_worker.AsyncSessionLocal") as mock_session:
        mock_ctx = AsyncMock()
        mock_ctx.begin = MagicMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.begin.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.execute = AsyncMock()

        from workers.dlq_worker import drain_loop
        try:
            await drain_loop(mock_redis)
        except asyncio.CancelledError:
            pass

    mock_ctx.execute.assert_called_once()


@pytest.mark.asyncio
async def test_drain_pushes_to_dlq_after_max_retries():
    payload = {"id": "fail-uuid", "card_id": "c2", "amount": 0.0,
               "fraud_probability": 0.0, "is_flagged": False,
               "threshold_used": 0.5, "latency_ms": 1.0, "created_at": "2026-06-08T00:00:00+00:00"}
    mock_redis = AsyncMock()
    mock_redis.blpop = AsyncMock(side_effect=[
        ("queue:prediction_logs", json.dumps(payload).encode()),
        asyncio.CancelledError,
    ])

    with patch("workers.dlq_worker.AsyncSessionLocal") as mock_session, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_ctx = AsyncMock()
        mock_ctx.begin = MagicMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.begin.return_value.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_ctx.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from workers.dlq_worker import drain_loop
        try:
            await drain_loop(mock_redis)
        except asyncio.CancelledError:
            pass

    mock_redis.rpush.assert_called_once()

