"""POST /predict — real-time inference route. Resolution: C-1, C-4, S-3."""
import math
import time
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from app.services.redis_cache import get_redis, push_card_amount, get_card_history_with_timestamps
from app.services.prediction_writer import enqueue_prediction_log
from app.services.config_service import config_service
from app.limiter import limiter

router = APIRouter()


class PredictRequest(BaseModel):
    card_id: str = Field(..., description="Unique card identifier")
    amount: float = Field(..., gt=0.0, description="Transaction amount")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")


class PredictResponse(BaseModel):
    prediction_id: str
    is_fraud: bool
    fraud_probability: float
    latency_ms: float


def prepare_prediction_features(
    amount: float,
    hour: int,
    history_items: list[tuple[float, float]],
    scaler: object,
    expected_features: list[str],
) -> pd.DataFrame:
    """Dynamically engineer and scale only the features required by the active model."""
    # 1. Parse card history
    amounts = [item[0] for item in history_items]
    timestamps = [item[1] for item in history_items]

    current_time = time.time()
    all_amounts = amounts + [amount]
    all_timestamps = timestamps + [current_time]

    # 2. Build pre-scaled features dictionary
    # V1-V28 are zero (unobserved in real-time)
    raw_feats = {f"V{i}": 0.0 for i in range(1, 29)}
    raw_feats["Amount"] = amount
    raw_feats["Amount_Log"] = math.log1p(amount)
    raw_feats["Amount_Normalized"] = (amount - 88.34961925093133) / 250.1201092401885

    # 3. Apply RobustScaler to V1-V28, Amount, Amount_Log, Amount_Normalized
    scaler_features = list(scaler.feature_names_in_)
    scaler_df = pd.DataFrame([raw_feats], columns=scaler_features)
    scaled_values = scaler.transform(scaler_df)[0]
    scaled_feats = dict(zip(scaler_features, scaled_values))

    # 4. Construct final feature dict dynamically based on expected_features
    final_row = {}

    # Pre-calculate temporal variables
    time_val = hour * 3600.0
    time_hours = float(hour)
    time_normalized = (time_val - 94813.85957508067) / 47488.14595456617
    time_hour = float(hour)
    time_hour_sin = math.sin(2 * math.pi * hour / 24)
    time_hour_cos = math.cos(2 * math.pi * hour / 24)

    # Pre-calculate Amount Bins
    amount_bin_low = 1.0 if 0 < amount <= 10 else 0.0
    amount_bin_medium = 1.0 if 10 < amount <= 100 else 0.0
    amount_bin_high = 1.0 if amount > 100 else 0.0
    amount_bin_nan = 1.0 if amount <= 0 else 0.0

    # Pre-calculate time diffs
    time_since_last = 0.0
    if len(all_timestamps) > 1:
        time_since_last = all_timestamps[-1] - all_timestamps[-2]

    # Pre-calculate rolling velocities
    avg_time_diff = 0.0
    if len(all_timestamps) > 1:
        diffs = np.diff(all_timestamps)
        avg_time_diff = float(np.mean(diffs[-10:]))

    tx_count_10 = float(min(len(all_amounts), 10))

    # Fill final_row based on expected features
    for col in expected_features:
        # Check if it is a scaled PCA or Amount feature
        if col in scaled_feats:
            final_row[col] = scaled_feats[col]
        # Check temporal features
        elif col == "Time":
            final_row[col] = time_val
        elif col == "Time_Hours":
            final_row[col] = time_hours
        elif col == "Time_Normalized":
            final_row[col] = time_normalized
        elif col == "Time_Hour":
            final_row[col] = time_hour
        elif col == "Time_Hour_Sin" or col == "hour_sin":
            final_row[col] = time_hour_sin
        elif col == "Time_Hour_Cos" or col == "hour_cos":
            final_row[col] = time_hour_cos
        # Check amount bins
        elif col == "Amount_Bin_Low":
            final_row[col] = amount_bin_low
        elif col == "Amount_Bin_Medium":
            final_row[col] = amount_bin_medium
        elif col == "Amount_Bin_High":
            final_row[col] = amount_bin_high
        elif col == "Amount_Bin_nan":
            final_row[col] = amount_bin_nan
        # Check rolling stats (windows 3, 5, 10)
        elif col.startswith("amt_mean_"):
            w = int(col.split("_")[-1])
            final_row[col] = float(np.mean(all_amounts[-w:]))
        elif col.startswith("amt_std_"):
            w = int(col.split("_")[-1])
            final_row[col] = float(np.std(all_amounts[-w:])) if len(all_amounts[-w:]) > 1 else 0.0
        elif col.startswith("amt_min_"):
            w = int(col.split("_")[-1])
            final_row[col] = float(np.min(all_amounts[-w:]))
        elif col.startswith("amt_max_"):
            w = int(col.split("_")[-1])
            final_row[col] = float(np.max(all_amounts[-w:]))
        elif col.startswith("amt_deviation_"):
            w = int(col.split("_")[-1])
            mean_val = np.mean(all_amounts[-w:])
            final_row[col] = amount - mean_val
        elif col.startswith("amt_zscore_"):
            w = int(col.split("_")[-1])
            mean_val = np.mean(all_amounts[-w:])
            std_val = np.std(all_amounts[-w:]) if len(all_amounts[-w:]) > 1 else 0.0
            final_row[col] = (amount - mean_val) / std_val if std_val > 0.0 else 0.0
        # Check velocity features
        elif col == "time_since_last":
            final_row[col] = time_since_last
        elif col == "tx_count_10":
            final_row[col] = tx_count_10
        elif col == "avg_time_diff":
            final_row[col] = avg_time_diff
        # Check cumulative features
        elif col == "amt_cumsum":
            final_row[col] = float(np.sum(all_amounts))
        elif col == "amt_cumcount":
            final_row[col] = float(len(all_amounts))
        elif col == "amt_cummean":
            final_row[col] = float(np.mean(all_amounts))
        # Extreme flags
        elif col == "amt_extremely_high":
            final_row[col] = 1.0 if amount > 1017.97 else 0.0
        elif col == "amt_extremely_low":
            final_row[col] = 1.0 if amount < 0.12 else 0.0
        elif col == "amt_overall_zscore":
            final_row[col] = (amount - 88.34961925093133) / 250.1201092401885
        # Polynomial squared terms
        elif col.endswith("_squared"):
            base_col = col.replace("_squared", "")
            val = scaled_feats.get(base_col, 0.0)
            final_row[col] = val ** 2
        # Interactions (amt_X_interaction)
        elif col.startswith("amt_") and col.endswith("_interaction"):
            base_col = col.replace("amt_", "").replace("_interaction", "")
            val = scaled_feats.get(base_col, 0.0)
            final_row[col] = scaled_feats.get("Amount", amount) * val
        elif col == "amt_ratio_3":
            mean_3 = np.mean(all_amounts[-3:])
            final_row[col] = amount / (mean_3 + 1e-8)
        else:
            final_row[col] = 0.0

    return pd.DataFrame([final_row], columns=expected_features)


@router.post("/predict", response_model=PredictResponse)
@limiter.limit("100/second")
async def predict(
    body: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictResponse:
    t0 = time.perf_counter()
    model = request.app.state.model
    scaler = request.app.state.scaler

    # Determine features expected by model signature
    if hasattr(model, "feature_name_"):
        expected_features = list(model.feature_name_)
    elif hasattr(model, "feature_name") and callable(model.feature_name):
        features_returned = model.feature_name()
        # Handle MagicMocks in tests
        if not isinstance(features_returned, list):
            expected_features = getattr(request.app.state, "feature_names", [])
        else:
            expected_features = list(features_returned)
    else:
        expected_features = getattr(request.app.state, "feature_names", [])

    # Fetch card history from Redis
    redis = await get_redis()
    history_items = await get_card_history_with_timestamps(redis, body.card_id)

    # Reconstruct exact features
    features_df = prepare_prediction_features(
        body.amount, body.hour, history_items, scaler, expected_features
    )

    # Execute LightGBM inference (Booster.predict returns list/array of probabilities)
    if hasattr(model, "best_iteration"):
        probs = model.predict(features_df, num_iteration=model.best_iteration)
    else:
        probs = model.predict(features_df)

    prob = float(probs[0])

    # Fetch dynamic decision threshold from DB (C-4)
    threshold = await config_service.get_float("shap_trigger_threshold", default=0.50)
    is_flagged = prob >= threshold
    latency_ms = (time.perf_counter() - t0) * 1000

    # Offload card amount caching and prediction WAL logging
    background_tasks.add_task(push_card_amount, redis, body.card_id, body.amount, time.time())
    
    # We pass the prediction ID to background tasks. To guarantee uniqueness and compliance,
    # we generate it immediately and return it in the response as well as logging it.
    import uuid
    prediction_id = str(uuid.uuid4())

    background_tasks.add_task(
        enqueue_prediction_log,
        card_id=body.card_id,
        amount=body.amount,
        fraud_probability=prob,
        is_flagged=is_flagged,
        threshold_used=threshold,
        latency_ms=latency_ms,
        prediction_id=prediction_id,
    )

    # In case transaction is flagged, trigger selective TreeSHAP explainer in Task 5
    if is_flagged:
        from app.services.shap_service import compute_and_store_shap
        background_tasks.add_task(
            compute_and_store_shap,
            prediction_id=prediction_id,
            model=model,
            features=features_df,
            feature_names=expected_features,
        )

    # Publish transaction event to live stream Pub/Sub channel
    from app.services.stream_publisher import publish_transaction, publish_alert
    background_tasks.add_task(
        publish_transaction,
        prediction_id=prediction_id,
        card_id=body.card_id,
        amount=body.amount,
        fraud_probability=prob,
        is_flagged=is_flagged,
        latency_ms=latency_ms,
    )

    # Trigger alerts for high confidence fraud (U-1) in alert worker
    alert_threshold = await config_service.get_float("alert_threshold", default=0.90)
    if prob >= alert_threshold:
        background_tasks.add_task(
            publish_alert,
            prediction_id=prediction_id,
            card_id=body.card_id,
            amount=body.amount,
            fraud_probability=prob,
        )

    return PredictResponse(
        prediction_id=prediction_id,
        is_fraud=is_flagged,
        fraud_probability=round(prob, 6),
        latency_ms=round(latency_ms, 3),
    )
