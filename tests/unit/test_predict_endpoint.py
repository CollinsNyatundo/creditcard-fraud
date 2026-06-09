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


@pytest.mark.asyncio
async def test_predict_with_all_engineered_features(mock_model):
    import time
    from unittest.mock import AsyncMock, MagicMock, patch
    from httpx import AsyncClient, ASGITransport
    from asgi_lifespan import LifespanManager

    # Mock scaler
    mock_scaler = MagicMock()
    mock_scaler.feature_names_in_ = ["V1", "Amount"]
    mock_scaler.transform.return_value = np.array([[0.5, 100.0]])

    # Mock redis & pipe
    mock_pipe = AsyncMock()
    mock_pipe.execute = AsyncMock(return_value=[1, 1, True])
    mock_redis = MagicMock()
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)
    
    # History with 2 items to trigger time differences and rolling calculations
    mock_history = [(50.0, time.time() - 100), (30.0, time.time() - 50)]
    
    expected_features = [
        "V1", "Amount", "Time", "Time_Hours", "Time_Normalized", "Time_Hour",
        "Time_Hour_Sin", "Time_Hour_Cos", "Amount_Bin_Low", "Amount_Bin_Medium",
        "Amount_Bin_High", "Amount_Bin_nan", "amt_mean_3", "amt_std_3", "amt_min_3",
        "amt_max_3", "amt_deviation_3", "amt_zscore_3", "time_since_last", "tx_count_10",
        "avg_time_diff", "amt_cumsum", "amt_cumcount", "amt_cummean", "amt_extremely_high",
        "amt_extremely_low", "amt_overall_zscore", "V1_squared", "amt_V1_interaction",
        "amt_ratio_3", "unknown_column"
    ]

    mock_model.feature_name_ = expected_features
    mock_model.predict.return_value = np.array([0.95])

    with patch("mlflow.lightgbm.load_model", return_value=mock_model), \
         patch("app.main.engine", new=_make_mock_engine()), \
         patch("app.routes.predict.get_redis", return_value=mock_redis), \
         patch("app.routes.predict.get_card_history_with_timestamps", AsyncMock(return_value=mock_history)), \
         patch("app.services.config_service.config_service.get_float",
               new_callable=AsyncMock, side_effect=lambda key, default=0.5: 0.5 if key == "shap_trigger_threshold" else 0.9), \
         patch("app.middleware.auth.AsyncSessionLocal") as mock_session:
         
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=("some-id",)))
        )
        
        from app.main import app

        async with LifespanManager(app):
            app.state.scaler = mock_scaler
            app.state.feature_names = expected_features
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/predict",
                    json={"card_id": "card_002", "amount": 105.0, "hour": 14},
                    headers={"X-API-Key": "test-key"},
                )
                
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == pytest.approx(0.95, abs=0.01)


@pytest.mark.asyncio
async def test_predict_fail_open_when_redis_down(mock_model):
    import time
    from unittest.mock import AsyncMock, MagicMock, patch
    from httpx import AsyncClient, ASGITransport
    from asgi_lifespan import LifespanManager

    # Mock scaler
    mock_scaler = MagicMock()
    mock_scaler.feature_names_in_ = ["V1", "Amount"]
    mock_scaler.transform.return_value = np.array([[0.5, 100.0]])
    
    expected_features = [
        "V1", "Amount", "Time", "Time_Hours", "Time_Normalized", "Time_Hour",
        "Time_Hour_Sin", "Time_Hour_Cos", "Amount_Bin_Low", "Amount_Bin_Medium",
        "Amount_Bin_High", "Amount_Bin_nan", "amt_mean_3", "amt_std_3", "amt_min_3",
        "amt_max_3", "amt_deviation_3", "amt_zscore_3", "time_since_last", "tx_count_10",
        "avg_time_diff", "amt_cumsum", "amt_cumcount", "amt_cummean", "amt_extremely_high",
        "amt_extremely_low", "amt_overall_zscore", "V1_squared", "amt_V1_interaction",
        "amt_ratio_3"
    ]

    mock_model.feature_name_ = expected_features
    mock_model.predict.return_value = np.array([0.95])

    # Force redis retrieval to fail
    with patch("mlflow.lightgbm.load_model", return_value=mock_model), \
         patch("app.main.engine", new=_make_mock_engine()), \
         patch("app.routes.predict.get_redis", side_effect=Exception("Redis connection refused")), \
         patch("app.services.config_service.config_service.get_float",
               new_callable=AsyncMock, side_effect=lambda key, default=0.5: 0.5 if key == "shap_trigger_threshold" else 0.9), \
         patch("app.middleware.auth.AsyncSessionLocal") as mock_session:
         
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=("some-id",)))
        )
        
        from app.main import app

        async with LifespanManager(app):
            app.state.scaler = mock_scaler
            app.state.feature_names = expected_features
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/predict",
                    json={"card_id": "card_003", "amount": 105.0, "hour": 14},
                    headers={"X-API-Key": "test-key"},
                )
                
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == pytest.approx(0.95, abs=0.01)


