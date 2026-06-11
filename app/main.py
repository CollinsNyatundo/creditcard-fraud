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

import hashlib
import os
import json
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


def validate_model_manifest() -> None:
    """Validate SHA-256 hashes of all files declared in models/model_manifest.json.

    Raises ValueError if a hash mismatch is found, causing startup failure.
    """
    manifest_path = "./models/model_manifest.json"
    if not os.path.exists(manifest_path):
        print("Warning: model_manifest.json not found. Skipping manifest validation.")
        return

    with open(manifest_path, "r") as f:
        try:
            manifest = json.load(f)
        except Exception as e:
            raise ValueError(f"Manifest Error: Failed to parse JSON manifest: {e}")

    artifacts = manifest.get("artifacts", {})
    for filename, expected_hash in artifacts.items():
        filepath = os.path.join("./models", filename)
        if not os.path.exists(filepath):
            raise ValueError(f"Manifest Error: Referenced file {filepath} not found on disk.")

        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"Manifest Cryptographic Mismatch: File {filepath} actual hash {actual_hash} "
                f"does not match expected manifest hash {expected_hash}."
            )
    print("Model manifest successfully verified (all SHA-256 hashes match).")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle manager.

    Startup:
        - Validate model manifest (SHA-256 verification)
        - Load LightGBM model once (with local cache fallback)
        - Load fitted RobustScaler preprocessor
        - Load behavior clusterer & configs
    """
    # Ensure IsotonicCalibratedBooster class is resolvable in __main__ namespace for pickle loading
    import sys
    try:
        from model.src.calibrate_probabilities import IsotonicCalibratedBooster
        import __main__
        __main__.IsotonicCalibratedBooster = IsotonicCalibratedBooster
    except ImportError:
        pass

    # 1. Cryptographic manifest validation
    try:
        validate_model_manifest()
    except Exception as e:
        print(f"CRITICAL: Model manifest verification failed: {e}")
        raise

    # 2. Load model with local cache fallback
    os.makedirs("./models/model_cache", exist_ok=True)
    fallback_cache_path = "./models/model_cache/calibrated_model_fallback.pkl"

    try:
        if os.path.exists("./models/calibrated_model.pkl"):
            app.state.model = joblib.load("./models/calibrated_model.pkl")
            print("Loaded calibrated model from calibrated_model.pkl")
            joblib.dump(app.state.model, fallback_cache_path)
        else:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            app.state.model = mlflow.lightgbm.load_model(settings.model_uri)
            print("Loaded baseline model from MLflow model URI")
            joblib.dump(app.state.model, fallback_cache_path)
    except Exception as e:
        print(f"Warning: Failed to load model from normal path: {e}. Trying local model cache fallback...")
        if os.path.exists(fallback_cache_path):
            try:
                app.state.model = joblib.load(fallback_cache_path)
                print(f"Successfully loaded fallback model from local cache: {fallback_cache_path}")
            except Exception as cache_err:
                print(f"CRITICAL: Failed to load fallback model from cache: {cache_err}")
                raise
        else:
            print("CRITICAL: MLflow/local model failed and no local fallback cache found.")
            raise

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
            active_key = threshold_config.get("active", "recall_target")
            if active_key in threshold_config:
                app.state.threshold = threshold_config[active_key]
            else:
                app.state.threshold = threshold_config.get("threshold", app.state.threshold)
            print(f"Resolved threshold value ({active_key}): {app.state.threshold}")
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
