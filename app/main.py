"""FastAPI application entry point.

Key design decisions implemented here:
- S-3: Model is loaded ONCE at application startup via the `lifespan` context manager,
  never per-request. MLflow is a build-time dependency only.
- C-3: Async DB engine is initialised at startup and disposed on shutdown.
- C-1: API key middleware is registered on all routes except health/docs.
"""
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
import mlflow.lightgbm
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.db.engine import engine
from app.limiter import limiter
from app.routes.predict import router as predict_router
from app.routes.stream import router as stream_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle manager.

    Startup:
        - Set MLflow tracking URI
        - Load LightGBM model once (resolves S-3: no per-request model fetch)
        - Set default decision threshold (overridden by DB config in Phase 2)
        - Load fitted RobustScaler preprocessor (for inference feature scaling)
        - Load canonical feature names list signature

    Shutdown:
        - Dispose async DB engine (closes all pool connections cleanly)
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    app.state.model = mlflow.lightgbm.load_model(settings.model_uri)
    app.state.threshold = settings.shap_trigger_threshold

    # Load scaling and feature name artifacts (C-2, C-3)
    app.state.scaler = joblib.load("./models/preprocessor.pkl")
    
    # Check if we have the optimized feature list
    feature_list_path = "./models/feature_list.json"
    if os.path.exists(feature_list_path):
        with open(feature_list_path, "r") as f:
            app.state.feature_names = json.load(f)
    else:
        with open("./models/feature_names.json", "r") as f:
            app.state.feature_names = json.load(f)["feature_names"]

    # Load behavior clusterer if present
    try:
        app.state.behavior_clusterer = joblib.load("./models/behavior_clusterer.pkl")
        with open("./models/behavior_clusterer_config.json", "r") as f:
            app.state.behavior_clusterer_config = json.load(f)
    except Exception as e:
        print(f"Warning: behavior clusterer not loaded: {e}")
        app.state.behavior_clusterer = None
        app.state.behavior_clusterer_config = None

    # Load optimal threshold and focal loss config if present
    try:
        with open("./models/optimal_threshold_v2.json", "r") as f:
            threshold_config = json.load(f)
            app.state.init_score = threshold_config.get("init_score", 0.0)
            app.state.is_focal_loss = threshold_config.get("is_focal_loss", False)
            app.state.threshold = threshold_config.get("threshold", app.state.threshold)
    except Exception as e:
        print(f"Warning: optimal_threshold_v2.json not loaded: {e}")
        app.state.init_score = 0.0
        app.state.is_focal_loss = False

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

# Set up rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Middleware (order matters: outermost wraps all routes) ---
from app.middleware.auth import APIKeyMiddleware  # noqa: E402

app.add_middleware(APIKeyMiddleware)


# --- Routes ---
app.include_router(predict_router)
app.include_router(stream_router)


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
