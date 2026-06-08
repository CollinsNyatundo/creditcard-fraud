"""Tests: FastAPI application startup and /health endpoint.

Verifies:
- Model is loaded at startup via the lifespan context manager (S-3).
- /health returns 200 with model_loaded=True when model is present.
- /health returns 200 with status=ok when model is loaded.
- /health is accessible without an API key.

Uses asgi-lifespan's LifespanManager to properly trigger the lifespan
context so that app.state.model gets populated during tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport


def _make_mock_engine() -> MagicMock:
    """Return a mock engine whose dispose() is an async no-op."""
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    return mock_engine


@pytest.fixture
def mock_mlflow_model() -> MagicMock:
    """A fake LightGBM model that satisfies `is not None` checks."""
    return MagicMock(name="FakeLGBMModel")


@pytest.mark.asyncio
async def test_health_returns_ok_when_model_loaded(mock_mlflow_model: MagicMock) -> None:
    """Health check returns status=ok and model_loaded=True after startup."""
    with (
        patch("mlflow.lightgbm.load_model", return_value=mock_mlflow_model),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


@pytest.mark.asyncio
async def test_health_accessible_without_api_key(mock_mlflow_model: MagicMock) -> None:
    """/health must not require an API key — it is in the auth whitelist."""
    with (
        patch("mlflow.lightgbm.load_model", return_value=mock_mlflow_model),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # No X-API-Key header — should still return 200
                response = await client.get("/health")

    assert response.status_code == 200
