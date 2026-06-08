"""Runtime configuration loaded from environment variables.

All settings are sourced from environment variables (or .env via python-dotenv).
See .env.example for the complete list of required keys.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # no-op if .env doesn't exist (e.g. in CI where env is injected)


class Settings:
    """Centralised application settings. All values come from environment."""

    database_url: str = os.environ.get("DATABASE_URL", "postgresql+asyncpg://fraud_user:dev_password@localhost:5432/fraud_db")
    redis_url: str = os.environ.get("REDIS_URL", "redis://:dev_password@localhost:6379/0")
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    model_uri: str = os.environ.get("MODEL_URI", "models:/fraud-lgbm/Production")
    api_key_header: str = os.environ.get("API_KEY_HEADER", "X-API-Key")
    stream_stale_threshold_seconds: int = int(
        os.environ.get("STREAM_STALE_THRESHOLD_SECONDS", "5")
    )
    stream_default_replay_seconds: int = int(
        os.environ.get("STREAM_DEFAULT_REPLAY_SECONDS", "60")
    )
    stream_max_replay_seconds: int = int(
        os.environ.get("STREAM_MAX_REPLAY_SECONDS", "600")
    )
    shap_trigger_threshold: float = float(
        os.environ.get("SHAP_TRIGGER_THRESHOLD", "0.50")
    )
    alert_threshold: float = float(os.environ.get("ALERT_THRESHOLD", "0.90"))
    alert_webhook_url: str = os.environ.get("ALERT_WEBHOOK_URL", "")
    alert_enabled: bool = os.environ.get("ALERT_ENABLED", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
