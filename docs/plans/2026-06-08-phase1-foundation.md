# Phase 1: Foundation & Infrastructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Stand up the full containerized service stack (PostgreSQL, Redis, MLflow, Metabase) and
lay the application skeleton with async DB engine, API key middleware, and canonical schema so that
Phases 2–4 can be built on a working, tested foundation.

**Architecture:** Docker Compose orchestrates all services. FastAPI loads the model at startup via
`lifespan`. Alembic manages all schema migrations. All DB access is async via `asyncpg` +
`SQLAlchemy[asyncio]`.

**Tech Stack:** FastAPI · asyncpg · SQLAlchemy[asyncio] · Alembic · Redis · PostgreSQL 16 ·
MLflow · Metabase · Docker Compose · pytest-asyncio · slowapi

**Design decisions reference:** `docs/design_decisions.md` — D-1, D-2, D-5, C-1, C-2, C-3, C-5, S-3

---

## Mandatory Gate

> ⚠️ **Do not begin Phase 2 until every task in this plan is marked `[x]`.**

---

### Task 1: Add new dependencies to `requirements.txt`

**Files:**
- Modify: `requirements.txt`

**Step 1: Append to `requirements.txt`**

```text
# Production backend
fastapi==0.115.5
uvicorn[standard]==0.34.0
asyncpg==0.30.0
sqlalchemy[asyncio]==2.0.41
alembic==1.16.1
redis[hiredis]==5.2.1
slowapi==0.1.9
httpx==0.27.0
python-dotenv==1.0.1
celery[redis]==5.5.2
prometheus-client==0.21.1

# Testing additions
pytest-asyncio==0.25.3
```

**Step 2: Verify**

```bash
pip install -r requirements.txt
# Expected: All packages resolve without conflict. Check lightgbm 4.6.0 + fastapi coexist.
```

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add production backend dependencies"
```

---

### Task 2: Create `.env.example` and verify `.env` is gitignored

**Files:**
- Create: `.env.example`
- Verify: `.gitignore` (already has `.env` — confirm, do not overwrite)

**Step 1: Create `.env.example`**

```bash
# .env.example — copy to .env and fill in real values. NEVER commit .env.
DATABASE_URL=postgresql+asyncpg://fraud_user:CHANGE_ME@localhost:5432/fraud_db
REDIS_URL=redis://:CHANGE_ME@localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5001
MODEL_URI=models:/fraud-lgbm/Production
API_KEY_HEADER=X-API-Key
STREAM_STALE_THRESHOLD_SECONDS=5
STREAM_DEFAULT_REPLAY_SECONDS=60
STREAM_MAX_REPLAY_SECONDS=600
SHAP_TRIGGER_THRESHOLD=0.50
ALERT_THRESHOLD=0.90
ALERT_WEBHOOK_URL=https://hooks.slack.com/CHANGE_ME
ALERT_ENABLED=false
```

**Step 2: Verify gitignore**

```bash
grep -n "^\.env$" .gitignore
# Expected: shows line number — .env is gitignored
```

**Step 3: Commit**

```bash
git add .env.example
git commit -m "chore: add .env.example with all required config keys"
```

---

### Task 3: Expand `docker-compose.yml` with all services

**Files:**
- Modify: `docker-compose.yml` (replace current single-container file)

**Step 1: Write the new compose file**

```yaml
version: '3.8'

services:
  api:
    build: .
    image: creditcard-fraud-api:latest
    container_name: fraud_api
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:16-alpine
    container_name: fraud_postgres
    environment:
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-dev_password}
      POSTGRES_DB: fraud_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud_user -d fraud_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: fraud_redis
    command: redis-server --requirepass ${REDIS_PASSWORD:-dev_password}
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD:-dev_password}", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.3
    container_name: fraud_mlflow
    ports:
      - "5001:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://fraud_user:${POSTGRES_PASSWORD:-dev_password}@postgres:5432/fraud_db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      mlflow server
        --backend-store-uri postgresql://fraud_user:${POSTGRES_PASSWORD:-dev_password}@postgres:5432/fraud_db
        --default-artifact-root /mlflow/artifacts
        --host 0.0.0.0

  metabase:
    image: metabase/metabase:v0.52.4
    container_name: fraud_metabase
    ports:
      - "3000:3000"
    environment:
      MB_DB_TYPE: postgres
      MB_DB_DBNAME: fraud_db
      MB_DB_PORT: 5432
      MB_DB_USER: fraud_user
      MB_DB_PASS: ${POSTGRES_PASSWORD:-dev_password}
      MB_DB_HOST: postgres
    depends_on:
      postgres:
        condition: service_healthy

  dlq_worker:
    build: .
    container_name: fraud_dlq_worker
    env_file: .env
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    command: python -m workers.dlq_worker

  alert_worker:
    build: .
    container_name: fraud_alert_worker
    env_file: .env
    depends_on:
      redis:
        condition: service_healthy
    command: python -m workers.alert_worker

volumes:
  postgres_data:
  mlflow_artifacts:
```

**Step 2: Verify compose parses**

```bash
docker compose config
# Expected: merged config printed with no errors
```

**Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: expand docker-compose with postgres, redis, mlflow, metabase, workers"
```

---

### Task 4: Create FastAPI app skeleton with `lifespan` and async DB engine

**Files:**
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/db/engine.py`
- Create: `app/db/__init__.py`
- Create: `app/config.py`

**Step 1: Write `app/config.py`**

```python
"""Runtime configuration loaded from environment variables."""
import os
from functools import lru_cache


class Settings:
    database_url: str = os.environ["DATABASE_URL"]
    redis_url: str = os.environ["REDIS_URL"]
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    model_uri: str = os.environ.get("MODEL_URI", "models:/fraud-lgbm/Production")
    api_key_header: str = os.environ.get("API_KEY_HEADER", "X-API-Key")
    stream_stale_threshold_seconds: int = int(os.environ.get("STREAM_STALE_THRESHOLD_SECONDS", "5"))
    stream_default_replay_seconds: int = int(os.environ.get("STREAM_DEFAULT_REPLAY_SECONDS", "60"))
    stream_max_replay_seconds: int = int(os.environ.get("STREAM_MAX_REPLAY_SECONDS", "600"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Step 2: Write `app/db/engine.py`**

```python
"""Async SQLAlchemy engine and session factory. Resolution: C-3."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=5,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

**Step 3: Write `app/main.py`**

```python
"""FastAPI application entry point. Model loaded once at startup (S-3)."""
from contextlib import asynccontextmanager
import mlflow.lightgbm
from fastapi import FastAPI
from app.config import get_settings
from app.db.engine import engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model ONCE at startup — never per-request
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    app.state.model = mlflow.lightgbm.load_model(settings.model_uri)
    app.state.threshold = 0.5  # overridden by DB config service in Phase 2
    yield
    await engine.dispose()


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/health")
async def health():
    model_loaded = getattr(app.state, "model", None) is not None
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded}
```

**Step 4: Write the failing test**

Create `tests/unit/test_app_startup.py`:

```python
"""Tests: FastAPI app startup and health endpoint."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_mlflow_model():
    return MagicMock()


@pytest.mark.asyncio
async def test_health_returns_ok_when_model_loaded(mock_mlflow_model):
    with patch("mlflow.lightgbm.load_model", return_value=mock_mlflow_model), \
         patch("app.db.engine.engine.dispose", new_callable=AsyncMock):
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True
```

**Step 5: Run test — expect PASS**

```bash
pytest tests/unit/test_app_startup.py -v
# Expected: PASSED
```

**Step 6: Commit**

```bash
git add app/ tests/unit/test_app_startup.py
git commit -m "feat: add fastapi app skeleton with lifespan model loading and health endpoint"
```

---

### Task 5: Create Alembic migrations — canonical schema

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/001_canonical_schema.py`
- Create: `alembic/versions/002_seed_feature_explanations.py`

**Step 1: Initialise Alembic**

```bash
alembic init alembic
# Creates alembic.ini and alembic/ directory
```

**Step 2: Update `alembic/env.py` — point at async engine**

Replace `run_migrations_online` with:

```python
import asyncio
from logging.config import fileConfig
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
from app.config import get_settings

config = context.config
fileConfig(config.config_file_name)
settings = get_settings()


def run_migrations_online():
    connectable = create_async_engine(settings.database_url)

    async def run():
        async with connectable.connect() as connection:
            await connection.run_sync(context.run_migrations)
        await connectable.dispose()

    asyncio.run(run())


run_migrations_online()
```

**Step 3: Write `alembic/versions/001_canonical_schema.py`**

```python
"""001: canonical schema — predictions, shap_explanations, system_config, api_keys.

Revision ID: 001
Revises: 
Create Date: 2026-06-08
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.create_table('predictions',
        sa.Column('id', UUID(), server_default=sa.text('gen_random_uuid()'), primary_key=True),
        sa.Column('card_id', sa.Text(), nullable=False),
        sa.Column('amount', sa.Numeric(12, 2), nullable=False),
        sa.Column('fraud_probability', sa.Float(), nullable=False),
        sa.Column('is_flagged', sa.Boolean(), nullable=False),
        sa.Column('threshold_used', sa.Float(), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.text('NOW()'), nullable=False),
    )
    op.create_index('idx_predictions_card_id', 'predictions', ['card_id'])
    op.create_index('idx_predictions_created_at', 'predictions', [sa.text('created_at DESC')])

    op.create_table('shap_explanations',
        sa.Column('id', UUID(), server_default=sa.text('gen_random_uuid()'), primary_key=True),
        sa.Column('prediction_id', UUID(), sa.ForeignKey('predictions.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('feature_name', sa.Text(), nullable=False),
        sa.Column('shap_value', sa.Float(), nullable=False),
        sa.Column('human_readable', sa.Text()),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.text('NOW()'), nullable=False),
    )
    op.create_index('idx_shap_prediction_id', 'shap_explanations', ['prediction_id'])

    op.create_table('system_config',
        sa.Column('key', sa.Text(), primary_key=True),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.text('NOW()'), nullable=False),
    )
    op.bulk_insert(
        sa.table('system_config',
                 sa.column('key', sa.Text()),
                 sa.column('value', sa.Text())),
        [
            {'key': 'shap_trigger_threshold', 'value': '0.50'},
            {'key': 'alert_threshold', 'value': '0.90'},
            {'key': 'alert_enabled', 'value': 'false'},
            {'key': 'alert_webhook_url', 'value': ''},
        ]
    )

    op.create_table('api_keys',
        sa.Column('id', UUID(), server_default=sa.text('gen_random_uuid()'), primary_key=True),
        sa.Column('key_hash', sa.Text(), unique=True, nullable=False),
        sa.Column('label', sa.Text()),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.text('NOW()'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
    )

    op.create_table('feature_explanations',
        sa.Column('feature_name', sa.Text(), primary_key=True),
        sa.Column('human_readable', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.text('NOW()'), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('feature_explanations')
    op.drop_table('api_keys')
    op.drop_table('system_config')
    op.drop_index('idx_shap_prediction_id', 'shap_explanations')
    op.drop_table('shap_explanations')
    op.drop_index('idx_predictions_created_at', 'predictions')
    op.drop_index('idx_predictions_card_id', 'predictions')
    op.drop_table('predictions')
```

**Step 4: Write `alembic/versions/002_seed_feature_explanations.py`**

```python
"""002: seed feature_explanations with V1–V28 PCA mappings.

Revision ID: 002
Revises: 001
Create Date: 2026-06-08
"""
from alembic import op
import sqlalchemy as sa

revision = '002'
down_revision = '001'


_MAPPINGS = [
    ('V1',  'Anonymized PCA component 1 — transaction behavioural pattern'),
    ('V2',  'Anonymized PCA component 2 — merchant category signal'),
    ('V3',  'Anonymized PCA component 3 — geographic velocity signal'),
    ('V4',  'Anonymized PCA component 4 — device fingerprint signal'),
    ('V5',  'Anonymized PCA component 5 — session duration signal'),
    ('V6',  'Anonymized PCA component 6 — spending category signal'),
    ('V7',  'Anonymized PCA component 7 — weekend activity pattern'),
    ('V8',  'Anonymized PCA component 8 — card age signal'),
    ('V9',  'Anonymized PCA component 9 — PIN vs. contactless ratio'),
    ('V10', 'Anonymized PCA component 10 — cross-border transaction signal'),
    ('V11', 'Anonymized PCA component 11 — high-value purchase signal'),
    ('V12', 'Anonymized PCA component 12 — refund pattern signal'),
    ('V13', 'Anonymized PCA component 13 — ATM usage ratio'),
    ('V14', 'Anonymized PCA component 14 — online vs. in-person ratio'),
    ('V15', 'Anonymized PCA component 15 — cashback frequency signal'),
    ('V16', 'Anonymized PCA component 16 — recurring payment pattern'),
    ('V17', 'Recent withdrawal amount is abnormally high'),
    ('V18', 'Anonymized PCA component 18 — velocity burst signal'),
    ('V19', 'Anonymized PCA component 19 — low-value micro-transaction signal'),
    ('V20', 'Anonymized PCA component 20 — after-hours activity signal'),
    ('V21', 'Anonymized PCA component 21 — multi-merchant session signal'),
    ('V22', 'Anonymized PCA component 22 — balance depletion rate'),
    ('V23', 'Anonymized PCA component 23 — card sharing signal'),
    ('V24', 'Anonymized PCA component 24 — spending volatility signal'),
    ('V25', 'Anonymized PCA component 25 — dormancy-then-burst signal'),
    ('V26', 'Anonymized PCA component 26 — terminal re-use pattern'),
    ('V27', 'Anonymized PCA component 27 — loyalty bypass signal'),
    ('V28', 'Anonymized PCA component 28 — chargeback history signal'),
    ('Amount', 'Raw transaction amount in USD'),
    ('hour_sin', 'Transaction time of day (sine cyclical encoding)'),
    ('hour_cos', 'Transaction occurred during non-business / sleep hours'),
    ('amount_log', 'Log-scaled transaction amount'),
    ('rolling_mean_amount', 'Rolling mean of last 10 transaction amounts for this card'),
    ('rolling_std_amount', 'Rolling standard deviation of last 10 amounts — spending volatility'),
]


def upgrade() -> None:
    table = sa.table('feature_explanations',
                     sa.column('feature_name', sa.Text()),
                     sa.column('human_readable', sa.Text()))
    op.bulk_insert(table, [{'feature_name': k, 'human_readable': v} for k, v in _MAPPINGS])


def downgrade() -> None:
    op.execute("DELETE FROM feature_explanations")
```

**Step 5: Run migrations against a local test DB**

```bash
# Spin up postgres only
docker compose up postgres -d

# Run migrations
DATABASE_URL=postgresql+asyncpg://fraud_user:dev_password@localhost:5432/fraud_db \
  alembic upgrade head

# Verify tables exist
docker compose exec postgres psql -U fraud_user -d fraud_db \
  -c "\dt" \
  -c "SELECT COUNT(*) FROM feature_explanations;"
# Expected: 5 tables listed; feature_explanations count = 35
```

**Step 6: Commit**

```bash
git add alembic/ alembic.ini
git commit -m "feat: add alembic canonical schema + feature_explanations seed (001, 002)"
```

---

### Task 6: Add API key authentication middleware

**Files:**
- Create: `app/middleware/__init__.py`
- Create: `app/middleware/auth.py`
- Modify: `app/main.py` — register middleware
- Create: `tests/unit/test_auth_middleware.py`

**Step 1: Write `app/middleware/auth.py`**

```python
"""API key authentication middleware. Resolution: C-1."""
import hashlib
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import get_settings
from app.db.engine import AsyncSessionLocal
import sqlalchemy as sa

settings = get_settings()

UNAUTHENTICATED_PATHS = {"/health", "/docs", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in UNAUTHENTICATED_PATHS:
            return await call_next(request)

        api_key = request.headers.get(settings.api_key_header)
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sa.text(
                    "SELECT id FROM api_keys WHERE key_hash = :hash AND is_active = true"
                ),
                {"hash": key_hash},
            )
            row = result.fetchone()

        if not row:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return await call_next(request)
```

**Step 2: Register middleware in `app/main.py`**

Add after the `app = FastAPI(...)` line:

```python
from app.middleware.auth import APIKeyMiddleware
app.add_middleware(APIKeyMiddleware)
```

**Step 3: Write the failing test**

```python
# tests/unit/test_auth_middleware.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_unauthenticated_predict_returns_401():
    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.db.engine.engine.dispose", new_callable=AsyncMock), \
         patch("app.middleware.auth.AsyncSessionLocal") as mock_session:
        mock_session.return_value.__aenter__.return_value.execute = AsyncMock(
            return_value=MagicMock(fetchone=MagicMock(return_value=None))
        )
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json={})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_health_accessible_without_auth():
    with patch("mlflow.lightgbm.load_model", return_value=MagicMock()), \
         patch("app.db.engine.engine.dispose", new_callable=AsyncMock):
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_auth_middleware.py tests/unit/test_app_startup.py -v
# Expected: 3 PASSED
```

**Step 5: Commit**

```bash
git add app/middleware/ tests/unit/test_auth_middleware.py app/main.py
git commit -m "feat: add API key authentication middleware with SHA-256 key hashing"
```

---

### Task 7: Implement Redis feature cache service with LTRIM pattern

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/redis_cache.py`
- Create: `tests/unit/test_redis_cache.py`

**Step 1: Write `app/services/redis_cache.py`**

```python
"""Redis feature cache with LTRIM hard-cap. Resolution: S-2, D-5."""
import redis.asyncio as aioredis
from app.config import get_settings

settings = get_settings()

WINDOW_SIZE = 10
TTL_SECONDS = 86400  # 24 hours


async def get_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def push_card_amount(redis: aioredis.Redis, card_id: str, amount: float) -> None:
    """Push amount and hard-cap list at WINDOW_SIZE. Atomic pipeline."""
    key = f"card:{card_id}:history"
    async with redis.pipeline(transaction=True) as pipe:
        pipe.rpush(key, str(amount))
        pipe.ltrim(key, -WINDOW_SIZE, -1)
        pipe.expire(key, TTL_SECONDS)
        await pipe.execute()


async def get_card_history(redis: aioredis.Redis, card_id: str) -> list[float]:
    """Return up to WINDOW_SIZE recent amounts for a card."""
    key = f"card:{card_id}:history"
    raw = await redis.lrange(key, 0, -1)
    return [float(x) for x in raw]
```

**Step 2: Write failing test**

```python
# tests/unit/test_redis_cache.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.redis_cache import push_card_amount, get_card_history, WINDOW_SIZE


@pytest.mark.asyncio
async def test_push_uses_ltrim_pipeline():
    """Verify pipeline calls rpush, ltrim, expire — never raw rpush alone."""
    mock_redis = MagicMock()
    mock_pipe = AsyncMock()
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)

    await push_card_amount(mock_redis, "card_123", 42.50)

    mock_pipe.rpush.assert_called_once_with("card:card_123:history", "42.5")
    mock_pipe.ltrim.assert_called_once_with("card:card_123:history", -WINDOW_SIZE, -1)
    mock_pipe.expire.assert_called_once_with("card:card_123:history", 86400)
    mock_pipe.execute.assert_called_once()
```

**Step 3: Run tests**

```bash
pytest tests/unit/test_redis_cache.py -v
# Expected: PASSED
```

**Step 4: Commit**

```bash
git add app/services/redis_cache.py tests/unit/test_redis_cache.py
git commit -m "feat: add Redis feature cache service with atomic RPUSH+LTRIM hard-cap"
```

---

### Phase 1 Done When

- [ ] `requirements.txt` includes all backend + test dependencies
- [ ] `.env.example` committed, `.env` gitignored, CI checks for leaked secrets
- [ ] `docker compose config` runs with no errors
- [ ] `docker compose up postgres -d` + `alembic upgrade head` creates all 5 tables with correct indexes
- [ ] `SELECT COUNT(*) FROM feature_explanations` returns 35
- [ ] `pytest tests/unit/test_app_startup.py tests/unit/test_auth_middleware.py tests/unit/test_redis_cache.py -v` → all **PASSED**
- [ ] All 6 tasks committed with semantic commit messages

> ✅ **Gate passed → proceed to [`2026-06-08-phase2-inference-path.md`](2026-06-08-phase2-inference-path.md)**
