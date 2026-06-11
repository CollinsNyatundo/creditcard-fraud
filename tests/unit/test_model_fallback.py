import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from asgi_lifespan import LifespanManager


def _make_mock_engine() -> MagicMock:
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    return mock_engine


@pytest.mark.asyncio
async def test_lifespan_fallback_model_loading() -> None:
    """If MLflow and calibrated_model.pkl fail to load, lifespan must fall back to the model_cache pkl."""
    mock_model = MagicMock(name="FallbackModel")
    mock_scaler = MagicMock(name="Scaler")
    
    # We mock:
    # - validate_model_manifest to succeed
    # - os.path.exists to say calibrated_model.pkl doesn't exist, but fallback cache exists
    # - mlflow.lightgbm.load_model to fail (simulating outage)
    # - joblib.load to load scaler and mock_model
    
    def mock_exists(path):
        if "calibrated_model.pkl" in str(path):
            return False
        if "calibrated_model_fallback.pkl" in str(path):
            return True
        if "preprocessor.pkl" in str(path):
            return True
        return False

    def mock_load(path):
        if "preprocessor.pkl" in str(path):
            return mock_scaler
        if "calibrated_model_fallback.pkl" in str(path):
            return mock_model
        raise ValueError(f"Unexpected joblib load for path: {path}")

    with (
        patch("app.main.validate_model_manifest", return_value=None),
        patch("os.path.exists", side_effect=mock_exists),
        patch("mlflow.lightgbm.load_model", side_effect=Exception("MLflow connection error")),
        patch("joblib.load", side_effect=mock_load),
        patch("joblib.dump", return_value=None),
        patch("app.main.engine", new=_make_mock_engine()),
    ):
        from app.main import app  # noqa: PLC0415
        
        async with LifespanManager(app):
            # Lifecycle started successfully
            assert app.state.model == mock_model
            assert app.state.scaler == mock_scaler
