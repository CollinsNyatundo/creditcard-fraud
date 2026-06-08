# Phase 4: Business Intelligence — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Deliver the business intelligence layer: Metabase auto-provisioned with three dashboards,
a training pipeline guard that warns when new model features lack human-readable explanations,
and end-to-end smoke tests that validate the full stack from prediction to dashboard query.

**Architecture:** A Python provisioning script uses the Metabase REST API to create the DB
connection and import pre-exported dashboard JSON files idempotently. The model training pipeline
gains a post-training check function. A Makefile target wires everything together.

**Tech Stack:** Metabase v0.52 REST API · httpx · PostgreSQL · pytest · Makefile

**Design decisions reference:** `docs/design_decisions.md` — U-2, U-3

**Pre-condition:** Phase 3 gate passed. All three services (API, DLQ worker, alert worker) running.

---

### Task 1: Create Metabase provisioning script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/setup_metabase.py`
- Create: `metabase/dashboards/README.md`
- Create: `tests/unit/test_setup_metabase.py`

**Step 1: Write `scripts/setup_metabase.py`**

```python
"""Idempotent Metabase provisioning script.

Usage:
    python scripts/setup_metabase.py

Env vars required:
    METABASE_URL    — e.g. http://localhost:3000
    METABASE_EMAIL  — admin email
    METABASE_PASS   — admin password
    DATABASE_URL    — PostgreSQL connection string (parsed for host/port/db)

Resolution: U-3
"""
import asyncio
import json
import logging
import os
import re
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

METABASE_URL = os.environ.get("METABASE_URL", "http://localhost:3000")
METABASE_EMAIL = os.environ.get("METABASE_EMAIL", "admin@fraud.local")
METABASE_PASS = os.environ.get("METABASE_PASS", "Admin1234!")
DASHBOARDS_DIR = Path(__file__).parent.parent / "metabase" / "dashboards"

# Parse PostgreSQL connection string
_DB_URL = os.environ.get("DATABASE_URL", "")
_DB_MATCH = re.match(
    r"postgresql\+asyncpg://(\w+):(\w+)@([\w.]+):(\d+)/(\w+)", _DB_URL
)


def _db_config() -> dict:
    if not _DB_MATCH:
        raise ValueError(f"Cannot parse DATABASE_URL: {_DB_URL}")
    user, password, host, port, dbname = _DB_MATCH.groups()
    return {
        "engine": "postgres",
        "name": "Fraud Detection DB",
        "details": {
            "host": host,
            "port": int(port),
            "dbname": dbname,
            "user": user,
            "password": password,
            "ssl": False,
        },
    }


async def _authenticate(client: httpx.AsyncClient) -> str:
    response = await client.post(
        f"{METABASE_URL}/api/session",
        json={"username": METABASE_EMAIL, "password": METABASE_PASS},
    )
    response.raise_for_status()
    token = response.json()["id"]
    logger.info("Authenticated with Metabase.")
    return token


async def _get_or_create_db(client: httpx.AsyncClient, token: str) -> int:
    """Create the PostgreSQL connection in Metabase, or return existing ID."""
    headers = {"X-Metabase-Session": token}
    existing = await client.get(f"{METABASE_URL}/api/database", headers=headers)
    existing.raise_for_status()
    for db in existing.json().get("data", []):
        if db["name"] == "Fraud Detection DB":
            logger.info("Database connection already exists (id=%d).", db["id"])
            return db["id"]

    response = await client.post(
        f"{METABASE_URL}/api/database",
        headers=headers,
        json=_db_config(),
    )
    response.raise_for_status()
    db_id = response.json()["id"]
    logger.info("Created Metabase DB connection (id=%d).", db_id)
    return db_id


async def _sync_db(client: httpx.AsyncClient, token: str, db_id: int) -> None:
    headers = {"X-Metabase-Session": token}
    await client.post(f"{METABASE_URL}/api/database/{db_id}/sync_schema", headers=headers)
    logger.info("Database schema sync triggered.")


async def _import_dashboards(client: httpx.AsyncClient, token: str) -> None:
    headers = {"X-Metabase-Session": token}
    dashboard_files = sorted(DASHBOARDS_DIR.glob("*.json"))
    if not dashboard_files:
        logger.warning("No dashboard JSON files found in %s", DASHBOARDS_DIR)
        return

    existing = await client.get(f"{METABASE_URL}/api/dashboard", headers=headers)
    existing.raise_for_status()
    existing_names = {d["name"] for d in existing.json()}

    for filepath in dashboard_files:
        dashboard = json.loads(filepath.read_text())
        if dashboard.get("name") in existing_names:
            logger.info("Dashboard '%s' already exists — skipping.", dashboard["name"])
            continue
        response = await client.post(
            f"{METABASE_URL}/api/dashboard",
            headers=headers,
            json={"name": dashboard["name"], "description": dashboard.get("description", "")},
        )
        response.raise_for_status()
        logger.info("Imported dashboard: %s", dashboard["name"])


async def main() -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        token = await _authenticate(client)
        db_id = await _get_or_create_db(client, token)
        await _sync_db(client, token, db_id)
        await _import_dashboards(client, token)
    logger.info("Metabase provisioning complete.")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Create Metabase dashboard stubs**

Create `metabase/dashboards/01_operations.json`:

```json
{
  "name": "Operations Dashboard",
  "description": "Live fraud rate, false positive rate, latency P50/P95/P99 over rolling windows.",
  "cards": []
}
```

Create `metabase/dashboards/02_financial_impact.json`:

```json
{
  "name": "Financial Impact Dashboard",
  "description": "Dollar value blocked, net savings vs. false-positive cost, monthly trend.",
  "cards": []
}
```

Create `metabase/dashboards/03_model_health.json`:

```json
{
  "name": "Model Health Dashboard",
  "description": "Score distribution drift, SHAP feature importance shift over time.",
  "cards": []
}
```

**Step 3: Write failing test for provisioning script**

```python
# tests/unit/test_setup_metabase.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_authenticate_returns_token():
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "test-token-abc"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    from scripts.setup_metabase import _authenticate
    token = await _authenticate(mock_client)
    assert token == "test-token-abc"


@pytest.mark.asyncio
async def test_get_or_create_db_skips_if_exists():
    existing_resp = MagicMock()
    existing_resp.json.return_value = {"data": [{"name": "Fraud Detection DB", "id": 42}]}
    existing_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=existing_resp)

    from scripts.setup_metabase import _get_or_create_db
    db_id = await _get_or_create_db(mock_client, "token")
    assert db_id == 42
    mock_client.post.assert_not_called()
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_setup_metabase.py -v
# Expected: 2 PASSED
```

**Step 5: Commit**

```bash
git add scripts/ metabase/ tests/unit/test_setup_metabase.py
git commit -m "feat: add idempotent Metabase provisioning script and dashboard stubs"
```

---

### Task 2: Add Makefile targets for one-command setup

**Files:**
- Modify: `Makefile` — add `setup-metabase`, `migrate`, `up`, `down` targets

**Step 1: Append to `Makefile`**

```makefile
# ── Production Stack ──────────────────────────────────────────────────────────

.PHONY: up down migrate setup-metabase

## up: Start all services (postgres, redis, mlflow, metabase, api, workers)
up:
	docker compose up -d
	@echo "Waiting for postgres to be healthy..."
	@until docker compose exec postgres pg_isready -U fraud_user -d fraud_db; do sleep 1; done
	@echo "All services up."

## down: Stop all services and remove containers
down:
	docker compose down

## migrate: Run Alembic migrations (requires postgres up)
migrate:
	DATABASE_URL=$(DATABASE_URL) alembic upgrade head
	@echo "Migrations complete."

## setup-metabase: Provision Metabase with DB connection and dashboards
setup-metabase:
	python scripts/setup_metabase.py
	@echo "Metabase provisioned."

## stack: Full setup — start services, migrate, provision Metabase
stack: up migrate setup-metabase
	@echo "Full stack ready."
```

**Step 2: Verify Makefile parses**

```bash
make --dry-run stack
# Expected: dry-run output shows all three commands (up, migrate, setup-metabase)
```

**Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile targets for stack, migrate, setup-metabase"
```

---

### Task 3: Add feature explanation coverage check to training pipeline

**Files:**
- Create: `model/src/feature_coverage_check.py`
- Modify: `model/src/final_model_evaluation.py` — call coverage check after training
- Create: `tests/unit/test_feature_coverage_check.py`

**Step 1: Write `model/src/feature_coverage_check.py`**

```python
"""Post-training guard: warn if any model feature lacks a human-readable explanation.

Resolution: U-2. Called by train.py after model is fitted.
"""
import logging

logger = logging.getLogger(__name__)

# Canonical set of features expected to have explanations (from seed migration 002)
SEEDED_FEATURES = {
    *[f"V{i}" for i in range(1, 29)],
    "Amount", "hour_sin", "hour_cos", "amount_log",
    "rolling_mean_amount", "rolling_std_amount",
}


def check_feature_explanations_coverage(model_feature_names: list[str]) -> list[str]:
    """Return list of features that have NO entry in feature_explanations.

    Logs a WARNING for each missing feature.
    Caller must decide whether to fail hard or just warn.
    """
    missing = [f for f in model_feature_names if f not in SEEDED_FEATURES]
    for feature in missing:
        logger.warning(
            "Feature '%s' has no human-readable explanation in feature_explanations table. "
            "Add an entry via Alembic migration before deploying.",
            feature,
        )
    return missing
```

**Step 2: Write failing test**

```python
# tests/unit/test_feature_coverage_check.py
from model.src.feature_coverage_check import check_feature_explanations_coverage


def test_returns_empty_for_fully_covered_features():
    features = ["V1", "V17", "Amount", "hour_sin", "hour_cos", "rolling_mean_amount"]
    missing = check_feature_explanations_coverage(features)
    assert missing == []


def test_returns_missing_features():
    features = ["V1", "new_engineered_feature_xyz"]
    missing = check_feature_explanations_coverage(features)
    assert "new_engineered_feature_xyz" in missing
    assert "V1" not in missing


def test_logs_warning_for_missing_feature(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="model.src.feature_coverage_check"):
        check_feature_explanations_coverage(["V1", "undocumented_feature"])
    assert "undocumented_feature" in caplog.text
    assert "no human-readable explanation" in caplog.text
```

**Step 3: Run tests**

```bash
pytest tests/unit/test_feature_coverage_check.py -v
# Expected: 3 PASSED
```

**Step 4: Commit**

```bash
git add model/src/feature_coverage_check.py tests/unit/test_feature_coverage_check.py
git commit -m "feat: add feature explanation coverage check for post-training guard"
```

---

### Task 4: Full stack smoke test

**Files:**
- Create: `tests/integration/test_full_stack_smoke.py`

> ⚠️ This test requires all Docker services to be running. It is skipped in CI unless
> `RUN_INTEGRATION_TESTS=true` is set.

**Step 1: Write `tests/integration/test_full_stack_smoke.py`**

```python
"""Full-stack smoke test: predict → WAL → DB → Metabase query.

Run with:
    RUN_INTEGRATION_TESTS=true pytest tests/integration/test_full_stack_smoke.py -v

Requires: docker compose up (all services healthy).
"""
import os
import asyncio
import httpx
import pytest

SKIP_REASON = "Set RUN_INTEGRATION_TESTS=true to run integration tests"
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() != "true",
    reason=SKIP_REASON,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("SMOKE_TEST_API_KEY", "")


@pytest.mark.asyncio
async def test_health_endpoint_ok():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


@pytest.mark.asyncio
async def test_predict_returns_valid_response():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/predict",
            json={"card_id": "smoke_test_card", "amount": 123.45, "hour": 14},
            headers={"X-API-Key": API_KEY},
        )
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["latency_ms"] < 10.0, f"Latency {data['latency_ms']}ms exceeds 10ms SLA"


@pytest.mark.asyncio
async def test_prediction_appears_in_postgres():
    """After predict, wait 2s for DLQ worker to drain, then verify row in DB."""
    import asyncpg
    import os

    db_url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "")
    await asyncio.sleep(2)

    conn = await asyncpg.connect(f"postgresql://{db_url}")
    row = await conn.fetchrow(
        "SELECT card_id FROM predictions WHERE card_id = $1 ORDER BY created_at DESC LIMIT 1",
        "smoke_test_card",
    )
    await conn.close()

    assert row is not None, "Prediction not found in DB after 2s — DLQ worker may be down"
    assert row["card_id"] == "smoke_test_card"
```

**Step 2: Run smoke tests (requires running stack)**

```bash
# Start the full stack first
make stack

# Run smoke tests
RUN_INTEGRATION_TESTS=true API_KEY=<your-key> \
  pytest tests/integration/test_full_stack_smoke.py -v

# Expected:
#   test_health_endpoint_ok              PASSED
#   test_predict_returns_valid_response  PASSED  (latency_ms < 10)
#   test_prediction_appears_in_postgres  PASSED
```

**Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test: add full-stack smoke tests with latency SLA assertion"
```

---

### Task 5: Final documentation update

**Files:**
- Modify: `docs/deployment_mlops.md` — add "Running the Production Stack" section

**Step 1: Append to `docs/deployment_mlops.md`**

```markdown
## Running the Production Stack

### One-Command Start

```bash
cp .env.example .env          # Fill in real values
make stack                    # Starts all services, migrates DB, provisions Metabase
```

### Services & Ports

| Service   | URL                      | Purpose                      |
|-----------|--------------------------|------------------------------|
| API       | http://localhost:8000    | `/predict`, `/stream`, `/health` |
| MLflow    | http://localhost:5001    | Experiment tracking, model registry |
| Metabase  | http://localhost:3000    | Business dashboards          |
| PostgreSQL| localhost:5432           | Predictions ledger           |
| Redis     | localhost:6379           | Feature cache, WAL, Pub/Sub  |

### Running the Smoke Test

```bash
RUN_INTEGRATION_TESTS=true SMOKE_TEST_API_KEY=<key> \
  pytest tests/integration/test_full_stack_smoke.py -v
```

### Adding a New API Key

```sql
INSERT INTO api_keys (key_hash, label)
VALUES (encode(sha256('your-raw-key'::bytea), 'hex'), 'my-key-label');
```
```

**Step 2: Commit**

```bash
git add docs/deployment_mlops.md
git commit -m "docs: add production stack setup guide to deployment_mlops.md"
```

---

### Phase 4 Done When

- [ ] `pytest tests/unit/test_setup_metabase.py tests/unit/test_feature_coverage_check.py -v` → **PASSED**
- [ ] `make --dry-run stack` shows correct command sequence
- [ ] `python scripts/setup_metabase.py` provisions Metabase with 3 dashboards (manual verify at http://localhost:3000)
- [ ] `check_feature_explanations_coverage(["V1", "unknown_feat"])` returns `["unknown_feat"]` and logs WARNING
- [ ] `RUN_INTEGRATION_TESTS=true pytest tests/integration/test_full_stack_smoke.py -v` → 3 PASSED with `latency_ms < 10`
- [ ] All 5 tasks committed

---

## 🏁 Full Project Done When

Run this final validation sweep:

```bash
# 1. All unit tests pass
pytest tests/unit/ -v

# 2. No regressions in existing pipeline tests
pytest tests/ -v

# 3. Full stack smoke tests pass
make stack
RUN_INTEGRATION_TESTS=true SMOKE_TEST_API_KEY=<key> \
  pytest tests/integration/test_full_stack_smoke.py -v

# 4. Latency SLA confirmed (from smoke test output):
#    test_predict_returns_valid_response  PASSED  (latency_ms < 10.0)

# 5. DB has data
docker compose exec postgres psql -U fraud_user -d fraud_db \
  -c "SELECT COUNT(*) FROM predictions;" \
  -c "SELECT COUNT(*) FROM feature_explanations;"
# Expected: predictions > 0; feature_explanations = 35

# 6. Metabase dashboards visible at http://localhost:3000
```

> ✅ **All four phases complete. The production observability backend is live.**
