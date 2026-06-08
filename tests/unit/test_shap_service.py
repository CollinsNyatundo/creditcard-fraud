# tests/unit/test_shap_service.py
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_shap_writes_rows_to_db():
    mock_model = MagicMock()
    features = np.zeros((1, 33))
    feature_names = [f"V{i}" for i in range(1, 29)] + \
                    ["Amount", "hour_sin", "hour_cos", "rolling_mean", "rolling_std"]

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [np.ones(33) * 0.1]

    with patch("shap.TreeExplainer", return_value=mock_explainer), \
         patch("app.services.shap_service.AsyncSessionLocal") as mock_session:
        mock_ctx = AsyncMock()
        mock_ctx.begin = MagicMock()  # Avoid coroutine context manager mock error
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
        mock_ctx.begin.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.services.shap_service import compute_and_store_shap
        await compute_and_store_shap("pred-123", mock_model, features, feature_names)

    # execute called: once for name map, once for insert
    assert mock_ctx.execute.call_count == 2
