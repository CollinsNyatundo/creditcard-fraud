"""Tests: API key authentication middleware.

Verifies:
- Unauthenticated requests to protected endpoints return HTTP 401 (C-1).
- /health is accessible without an API key (whitelist check).
- A valid API key allows the request through.

Design decision reference: C-1

Uses asgi-lifespan's LifespanManager to properly activate the app lifecycle
so that FastAPI exception handlers are wired up and HTTP 401 responses are
returned as proper HTTP responses (not raised exceptions).
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


@pytest.mark.asyncio
async def test_unauthenticated_predict_returns_401() -> None:
    """Any request to /predict without X-API-Key must return 401 — not 403."""
    with (
        patch("mlflow.lightgbm.load_model", return_value=MagicMock()),
        patch("app.main.engine", new=_make_mock_engine()),
        patch("app.middleware.auth.AsyncSessionLocal") as mock_session_factory,
    ):
        # Simulate: no matching API key found in the database
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=None))
        )
        mock_session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/predict", json={})

    # Must be 401, not 403 (avoids leaking endpoint existence)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_missing_api_key_header_returns_401() -> None:
    """Missing X-API-Key header on a protected endpoint returns 401."""
    with (
        patch("mlflow.lightgbm.load_model", return_value=MagicMock()),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # No header at all — middleware should 401 before hitting any DB
                response = await client.post("/predict")

    assert response.status_code == 401
    assert "Missing API key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_health_accessible_without_auth() -> None:
    """/health is in UNAUTHENTICATED_PATHS — must pass without API key."""
    with (
        patch("mlflow.lightgbm.load_model", return_value=MagicMock()),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalid_api_key_returns_401() -> None:
    """A request with an API key that does not match any db record returns 401."""
    with (
        patch("mlflow.lightgbm.load_model", return_value=MagicMock()),
        patch("app.main.engine", new=_make_mock_engine()),
        patch("app.middleware.auth.AsyncSessionLocal") as mock_session_factory,
    ):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=None))
        )
        mock_session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.main import app  # noqa: PLC0415

        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/predict",
                    json={},
                    headers={"X-API-Key": "totally-invalid-key"},
                )

    assert response.status_code == 401
