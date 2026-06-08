# tests/unit/test_predict_endpoint.py
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager


def _make_mock_engine() -> MagicMock:
    """Return a mock engine whose dispose() is an async no-op."""
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    return mock_engine


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([0.92])
    return m


@pytest.mark.asyncio
async def test_predict_returns_fraud_flag(mock_model):
    mock_pipe = AsyncMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.ltrim = MagicMock()
    mock_pipe.expire = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[1, 1, True])

    mock_redis = MagicMock()
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_redis.lrange = AsyncMock(return_value=[])
    
    with patch("mlflow.lightgbm.load_model", return_value=mock_model), \
         patch("app.main.engine", new=_make_mock_engine()), \
         patch("app.routes.predict.get_redis", return_value=mock_redis), \
         patch("app.services.config_service.config_service.get_float",
               new_callable=AsyncMock, return_value=0.5), \
         patch("app.middleware.auth.AsyncSessionLocal") as mock_session:
        # Auth: valid key
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=("some-id",)))
        )
        from app.main import app
        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/predict",
                    json={"card_id": "card_001", "amount": 500.0, "hour": 3},
                    headers={"X-API-Key": "test-key"},
                )
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == pytest.approx(0.92, abs=0.01)
    assert "latency_ms" in data
