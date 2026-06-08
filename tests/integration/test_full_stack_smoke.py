# tests/integration/test_full_stack_smoke.py
import asyncio
import hashlib
import os
import secrets
import httpx
import pytest
import sqlalchemy as sa
from app.config import get_settings
from app.db.engine import AsyncSessionLocal

SKIP_REASON = "Set RUN_INTEGRATION_TESTS=true to run integration tests"
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() != "true",
    reason=SKIP_REASON,
)

settings = get_settings()
API_URL = os.environ.get("API_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def temp_api_key():
    """Seed a temporary API key in PostgreSQL and clean it up after tests."""
    raw_key = "smoke-test-" + secrets.token_hex(16)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                sa.text(
                    "INSERT INTO api_keys (key_hash, label, is_active) VALUES (:hash, 'smoke-test-key', true)"
                ),
                {"hash": key_hash},
            )

    yield raw_key

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                sa.text("DELETE FROM api_keys WHERE key_hash = :hash"),
                {"hash": key_hash},
            )


@pytest.mark.anyio
async def test_health_endpoint_ok():
    """Verify that the health check endpoint returns 200 and shows model is loaded."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_predict_returns_valid_response(temp_api_key):
    """Verify that `/predict` returns a valid score and meets latency SLA (<10ms)."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/predict",
            json={"card_id": "smoke_test_card", "amount": 123.45, "hour": 14},
            headers={settings.api_key_header: temp_api_key},
        )
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["latency_ms"] < 10.0, f"Latency {data['latency_ms']}ms exceeds 10ms SLA"


@pytest.mark.anyio
async def test_prediction_appears_in_postgres(temp_api_key):
    """Confirm the full prediction pipeline: API -> Redis WAL -> PG."""
    unique_card_id = "smoke-card-" + secrets.token_hex(8)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/predict",
            json={"card_id": unique_card_id, "amount": 99.99, "hour": 10},
            headers={settings.api_key_header: temp_api_key},
        )
    assert response.status_code == 200

    # Wait for the background worker to drain the Redis WAL and write to PG
    await asyncio.sleep(2.5)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa.text(
                "SELECT card_id, amount, is_flagged FROM predictions WHERE card_id = :card_id"
            ),
            {"card_id": unique_card_id},
        )
        row = result.fetchone()

    assert row is not None, f"Prediction for card {unique_card_id} was not persisted"
    assert row[0] == unique_card_id
    assert float(row[1]) == 99.99
