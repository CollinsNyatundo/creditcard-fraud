# tests/unit/test_config_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_get_returns_db_value_on_cache_miss():
    from app.services.config_service import ConfigService

    service = ConfigService(ttl=60)
    mock_row = MagicMock()
    mock_row.__getitem__ = MagicMock(return_value="0.75")

    with patch("app.services.config_service.AsyncSessionLocal") as mock_session:
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=mock_row))
        )
        result = await service.get_float("shap_trigger_threshold", default=0.5)

    assert result == 0.75


@pytest.mark.asyncio
async def test_get_returns_cached_value_on_cache_hit():
    from app.services.config_service import ConfigService
    import time

    service = ConfigService(ttl=60)
    service._cache["shap_trigger_threshold"] = ("0.80", time.monotonic() + 60)

    with patch("app.services.config_service.AsyncSessionLocal") as mock_session:
        result = await service.get_float("shap_trigger_threshold", default=0.5)
        mock_session.assert_not_called()

    assert result == 0.80
