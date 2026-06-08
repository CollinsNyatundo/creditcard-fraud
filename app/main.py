"""FastAPI application entry point.

Key design decisions implemented here:
- S-3: Model is loaded ONCE at application startup via the `lifespan` context manager,
  never per-request. MLflow is a build-time dependency only.
- C-3: Async DB engine is initialised at startup and disposed on shutdown.
- C-1: API key middleware is registered on all routes except health/docs.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import mlflow.lightgbm
from fastapi import FastAPI

from app.config import get_settings
from app.db.engine import engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle manager.

    Startup:
        - Set MLflow tracking URI
        - Load LightGBM model once (resolves S-3: no per-request model fetch)
        - Set default decision threshold (overridden by DB config in Phase 2)

    Shutdown:
        - Dispose async DB engine (closes all pool connections cleanly)
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    app.state.model = mlflow.lightgbm.load_model(settings.model_uri)
    app.state.threshold = settings.shap_trigger_threshold
    yield
    await engine.dispose()


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection with sub-10ms inference latency, "
        "async SHAP explanations, and live WebSocket monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- Middleware (order matters: outermost wraps all routes) ---
from app.middleware.auth import APIKeyMiddleware  # noqa: E402

app.add_middleware(APIKeyMiddleware)


# --- Routes ---

@app.get("/health", tags=["Ops"])
async def health() -> dict:
    """Health check endpoint.

    Returns 200 with model_loaded=True when the model is ready to serve.
    Returns 200 with model_loaded=False (status=degraded) if startup failed.
    This endpoint is intentionally unauthenticated (see auth middleware).
    """
    model_loaded: bool = getattr(app.state, "model", None) is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }
