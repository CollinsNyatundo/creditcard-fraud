"""Root conftest — sets environment variables before any app module is imported.

This prevents Settings() from raising errors when DATABASE_URL / REDIS_URL are
not present in the test environment. All values here are non-functional test
doubles; no real network connections are made in unit tests.
"""
import os

import pytest

# Set environment vars BEFORE importing any app modules so that lru_cache'd
# Settings() instances pick up the test values.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://:test@localhost:6379/0")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ.setdefault("MODEL_URI", "models:/fraud-lgbm/Production")
