# tests/unit/test_setup_metabase.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from scripts.setup_metabase import (
    _authenticate,
    _get_or_create_db,
    _get_or_create_card,
    _import_dashboards,
)


@pytest.mark.asyncio
async def test_authenticate_returns_token():
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "test-token-abc"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    token = await _authenticate(mock_client)
    assert token == "test-token-abc"
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_get_or_create_db_skips_if_exists():
    existing_resp = MagicMock()
    existing_resp.json.return_value = {"data": [{"name": "Fraud Detection DB", "id": 42}]}
    existing_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=existing_resp)

    db_id = await _get_or_create_db(mock_client, "token")
    assert db_id == 42
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_get_or_create_db_creates_if_not_exists():
    existing_resp = MagicMock()
    existing_resp.json.return_value = {"data": []}
    existing_resp.raise_for_status = MagicMock()

    create_resp = MagicMock()
    create_resp.json.return_value = {"id": 100}
    create_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=existing_resp)
    mock_client.post = AsyncMock(return_value=create_resp)

    db_id = await _get_or_create_db(mock_client, "token")
    assert db_id == 100
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_get_or_create_card_skips_if_exists():
    existing_resp = MagicMock()
    existing_resp.json.return_value = [{"name": "Live Fraud Rate (24h)", "id": 7}]
    existing_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=existing_resp)

    card_def = {
        "name": "Live Fraud Rate (24h)",
        "query": "SELECT * FROM predictions",
        "display": "scalar",
    }
    card_id = await _get_or_create_card(mock_client, "token", 1, card_def)
    assert card_id == 7
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_import_dashboards_creates_dashboard_and_cards():
    existing_dashboards_resp = MagicMock()
    existing_dashboards_resp.json.return_value = []
    existing_dashboards_resp.raise_for_status = MagicMock()

    create_dashboard_resp = MagicMock()
    create_dashboard_resp.json.return_value = {"id": 200}
    create_dashboard_resp.raise_for_status = MagicMock()

    dash_detail_resp = MagicMock()
    dash_detail_resp.json.return_value = {"ordered_cards": []}
    dash_detail_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    # Ordered mock responses for client.get and client.post calls
    mock_client.get.side_effect = [existing_dashboards_resp, dash_detail_resp]
    mock_client.post.side_effect = [create_dashboard_resp, MagicMock()]

    card_def = {
        "name": "Live Fraud Rate (24h)",
        "query": "SELECT 1",
        "display": "scalar",
    }

    with patch("scripts.setup_metabase.DASHBOARDS_DIR") as mock_dir, \
         patch("scripts.setup_metabase._get_or_create_card", new_callable=AsyncMock, return_value=77) as mock_get_card:
        mock_file = MagicMock()
        mock_file.read_text.return_value = '{"name": "Ops", "description": "Desc", "cards": [{"name": "Live Fraud Rate (24h)", "query": "SELECT 1", "display": "scalar"}]}'
        mock_dir.glob.return_value = [mock_file]

        await _import_dashboards(mock_client, "token", 1)

        mock_get_card.assert_called_once()
