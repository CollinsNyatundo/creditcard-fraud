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
    if hasattr(m, "predict_proba"):
        del m.predict_proba
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
            app.state.model = mock_model
            app.state.is_focal_loss = False
            app.state.init_score = 0.0
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
            app.state.model = mock_model
            app.state.is_focal_loss = False
            app.state.init_score = 0.0
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
            app.state.model = mock_model
            app.state.is_focal_loss = False
            app.state.init_score = 0.0
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




# ---------------------------------------------------------------------------
# New: Calibrated model path (P0 verification)
# ---------------------------------------------------------------------------

import pytest as _pytest
import numpy as _np
from unittest.mock import AsyncMock as _AsyncMock, MagicMock as _MagicMock, patch as _patch
from httpx import AsyncClient as _AsyncClient, ASGITransport as _ASGITransport
from asgi_lifespan import LifespanManager as _LifespanManager


@_pytest.mark.asyncio
async def test_predict_with_calibrated_model():
    """Verify that when app.state.model is an IsotonicCalibratedBooster,
    the /predict endpoint uses predict_proba (not predict) and returns
    the correct class-1 probability."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from model.src.calibrate_probabilities import IsotonicCalibratedBooster

    calibrated_mock = _MagicMock(spec=IsotonicCalibratedBooster)
    calibrated_mock.predict_proba.return_value = _np.array([[0.12, 0.88]])
    calibrated_mock.feature_name_ = [
        "V1", "Amount", "Time", "Time_Hours", "Time_Normalized",
        "Time_Hour", "Time_Hour_Sin", "Time_Hour_Cos",
        "Amount_Bin_Low", "Amount_Bin_Medium", "Amount_Bin_High", "Amount_Bin_nan",
    ]

    mock_scaler = _MagicMock()
    mock_scaler.feature_names_in_ = ["V1", "Amount"]
    mock_scaler.transform.return_value = _np.array([[0.5, 100.0]])

    mock_pipe = _AsyncMock()
    mock_pipe.execute = _AsyncMock(return_value=[1, 1, True])
    mock_redis = _MagicMock()
    mock_redis.pipeline.return_value.__aenter__ = _AsyncMock(return_value=mock_pipe)
    mock_redis.pipeline.return_value.__aexit__ = _AsyncMock(return_value=False)
    mock_redis.lrange = _AsyncMock(return_value=[])

    def _make_mock_engine_local():
        me = _MagicMock()
        me.dispose = _AsyncMock()
        return me

    with _patch("mlflow.lightgbm.load_model", return_value=calibrated_mock), \
         _patch("app.main.engine", new=_make_mock_engine_local()), \
         _patch("app.routes.predict.get_redis", return_value=mock_redis), \
         _patch("app.services.config_service.config_service.get_float",
                new_callable=_AsyncMock, return_value=0.5), \
         _patch("app.middleware.auth.AsyncSessionLocal") as mock_session:

        mock_session.return_value.__aenter__.return_value.execute = _AsyncMock(
            return_value=_MagicMock(fetchone=_MagicMock(return_value=("some-id",)))
        )
        from app.main import app

        async with _LifespanManager(app):
            app.state.model = calibrated_mock
            app.state.scaler = mock_scaler
            app.state.is_focal_loss = False
            app.state.init_score = 0.0
            app.state.feature_names = calibrated_mock.feature_name_
            async with _AsyncClient(transport=_ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/predict",
                    json={"card_id": "card_calib_001", "amount": 200.0, "hour": 10},
                    headers={"X-API-Key": "test-key"},
                )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == _pytest.approx(0.88, abs=0.01)
    calibrated_mock.predict_proba.assert_called_once()


# ---------------------------------------------------------------------------
# New: Latency regression guard (P1 / R8 - p95 must be under 10ms)
# ---------------------------------------------------------------------------

def test_inference_latency_p95_under_10ms():
    """R8: single-row feature prep + model inference p95 must stay under 10ms.

    Runs 1000 consecutive in-process predict cycles with no I/O.
    """
    import time
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from app.routes.predict import prepare_prediction_features

    mock_scaler = _MagicMock()
    mock_scaler.feature_names_in_ = ["V1", "Amount", "Amount_Log", "Amount_Normalized"]
    mock_scaler.transform.return_value = _np.array([[0.0, 1.0, 0.1, 0.2]])

    mock_model = _MagicMock()
    mock_model.predict.return_value = _np.array([0.3])

    expected_features = [
        "V1", "Amount", "Time", "Time_Hours", "Time_Normalized",
        "Time_Hour", "Time_Hour_Sin", "Time_Hour_Cos",
        "Amount_Bin_Low", "Amount_Bin_Medium", "Amount_Bin_High",
        "Amount_Bin_nan", "amt_mean_3", "amt_std_3", "amt_min_3",
        "amt_max_3", "amt_deviation_3", "amt_zscore_3",
        "time_since_last", "tx_count_10", "avg_time_diff",
        "amt_cumsum", "amt_cumcount", "amt_cummean",
        "amt_extremely_high", "amt_extremely_low", "amt_overall_zscore",
    ]
    history = [(50.0, time.time() - 60), (30.0, time.time() - 30)]

    N = 1000
    latencies = []
    for _ in range(N):
        t0 = time.perf_counter()
        features_df = prepare_prediction_features(
            amount=100.0,
            hour=14,
            history_items=history,
            scaler=mock_scaler,
            expected_features=expected_features,
        )
        mock_model.predict(features_df)
        latencies.append((time.perf_counter() - t0) * 1000)

    p95 = float(_np.percentile(latencies, 95))
    gate = 10.0
    assert p95 < gate, (
        f"R8 LATENCY SLA VIOLATED: p95 = {p95:.2f}ms (gate: {gate}ms). "
        "Investigate the feature preparation or model predict path."
    )
