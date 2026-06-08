# tests/unit/test_stream_endpoint.py
import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport


def _make_mock_engine() -> MagicMock:
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    return mock_engine


@pytest.fixture
def mock_redis():
    mock_redis = AsyncMock()
    # Mock token lookup
    mock_redis.get = AsyncMock(return_value="test-api-key-id")
    mock_redis.delete = AsyncMock(return_value=1)

    # Mock Pub/Sub listen channel
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.close = AsyncMock()

    async def listen_gen():
        # First message is standard subscription info
        yield {"type": "subscribe", "channel": b"fraud:stream", "data": 1}
        # Second message is a live transaction
        yield {
            "type": "message",
            "channel": b"fraud:stream",
            "data": json.dumps({
                "prediction_id": "live-pred-id",
                "card_id": "c1",
                "amount": 120.0,
                "fraud_probability": 0.05,
                "is_flagged": False,
                "latency_ms": 2.5,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        }

    mock_pubsub.listen = listen_gen
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
    return mock_redis


@pytest.mark.asyncio
async def test_generate_stream_token_success():
    """POST /auth/stream-token returns token response when authenticated."""
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)

    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.main.engine", new=_make_mock_engine()), \
         patch("app.routes.stream.get_redis", return_value=mock_redis), \
         patch("app.middleware.auth.AsyncSessionLocal") as mock_session:
        # Auth: valid API key
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=("api-key-uuid",)))
        )

        from app.main import app
        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/auth/stream-token",
                    headers={"X-API-Key": "test-key"},
                )

    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    mock_redis.setex.assert_called_once()
    assert mock_redis.setex.call_args[0][1] == 60  # 60s TTL


@pytest.mark.asyncio
async def test_generate_stream_token_unauthenticated():
    """POST /auth/stream-token returns 401 when missing API key."""
    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.main.engine", new=_make_mock_engine()):
        from app.main import app
        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/auth/stream-token")

    assert response.status_code == 401


def test_stream_rejects_missing_token():
    """WebSocket closes with error on missing token."""
    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.main.engine", new=_make_mock_engine()):
        from app.main import app
        client = TestClient(app)
        with client.websocket_connect("/stream") as ws:
            msg = ws.receive_text()
            data = json.loads(msg)
            assert "Missing stream token" in data["error"]


def test_stream_rejects_invalid_token():
    """WebSocket closes with error on invalid/expired token."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)  # Token not found

    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.main.engine", new=_make_mock_engine()), \
         patch("app.routes.stream.get_redis", return_value=mock_redis):
        from app.main import app
        client = TestClient(app)
        with client.websocket_connect("/stream?token=badtoken") as ws:
            msg = ws.receive_text()
            data = json.loads(msg)
            assert "Invalid or expired stream token" in data["error"]


def test_stream_rejects_bad_since_format(mock_redis):
    """WebSocket closes with error on invalid ISO timestamp for since parameter."""
    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.main.engine", new=_make_mock_engine()), \
         patch("app.routes.stream.get_redis", return_value=mock_redis):
        from app.main import app
        client = TestClient(app)
        with client.websocket_connect("/stream?token=goodtoken&since=not-a-date") as ws:
            msg = ws.receive_text()
            data = json.loads(msg)
            assert "invalid since format" in data["error"]


def test_stream_successful_flow(mock_redis):
    """WebSocket streams history replay then publishes live events."""
    mock_rows = [
        ("uuid-1", "card_1", 50.0, 0.01, False, 1.2, datetime.now(timezone.utc)),
    ]
    mock_result = MagicMock()
    mock_result.fetchall.return_value = mock_rows

    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.main.engine", new=_make_mock_engine()), \
         patch("app.routes.stream.get_redis", return_value=mock_redis), \
         patch("app.routes.stream.AsyncSessionLocal") as mock_session:
        mock_ctx = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.execute = AsyncMock(return_value=mock_result)

        from app.main import app
        client = TestClient(app)
        with client.websocket_connect("/stream?token=goodtoken&since=2026-06-08T00:00:00Z") as ws:
            # 1. Verify replay packet
            msg_replay = ws.receive_text()
            data_replay = json.loads(msg_replay)
            assert data_replay["type"] == "replay"
            assert data_replay["card_id"] == "card_1"
            assert data_replay["amount"] == 50.0

            # 2. Verify live packet
            msg_live = ws.receive_text()
            data_live = json.loads(msg_live)
            assert data_live["type"] == "live"
            assert data_live["prediction_id"] == "live-pred-id"
            assert data_live["amount"] == 120.0
