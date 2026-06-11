"""POST /predict — real-time inference route. Resolution: C-1, C-4, S-3."""
import math
import time
import uuid
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from scipy import special

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
    behavior_clusterer: object = None,
    behavior_clusterer_config: dict = None,
    redis_failed: bool = False,
    features_hash: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Dynamically engineer and scale only the features required by the active model."""
    # 1. Parse card history
    current_time = time.time()
    
    if features_hash:
        try:
            raw_amounts = features_hash.get("amounts", "")
            raw_timestamps = features_hash.get("timestamps", "")
            amounts = [float(x) for x in raw_amounts.split(",") if x]
            timestamps = [float(x) for x in raw_timestamps.split(",") if x]
        except Exception:
            # Fallback if hash is corrupt
            amounts = [item[0] for item in history_items]
            timestamps = [item[1] for item in history_items]
            features_hash = None
    else:
        amounts = [item[0] for item in history_items]
        timestamps = [item[1] for item in history_items]

    all_amounts = amounts + [amount]
    all_timestamps = timestamps + [current_time]

    # Pre-parse aggregates from Hash if available
    hash_aggregates = {}
    if features_hash:
        try:
            for w in [2, 4, 9]:
                hash_aggregates[w] = {
                    "count": int(features_hash.get(f"count_{w}", 0)),
                    "sum": float(features_hash.get(f"sum_{w}", 0.0)),
                    "sum_sq": float(features_hash.get(f"sum_sq_{w}", 0.0)),
                    "min": float(features_hash.get(f"min_{w}", 0.0)),
                    "max": float(features_hash.get(f"max_{w}", 0.0)),
                }
        except Exception:
            features_hash = None

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

    # Pre-calculate time diffs and rolling velocities
    time_since_last = 0.0
    avg_time_diff = 0.0
    if features_hash:
        try:
            if len(all_timestamps) > 1:
                time_since_last = all_timestamps[-1] - all_timestamps[-2]
                diffs = np.diff(all_timestamps)
                avg_time_diff = float(np.mean(diffs[-10:]))
            count_9 = int(features_hash.get("count_9", 0))
            tx_count_10 = float(min(count_9 + 1, 10))
        except Exception:
            if len(all_timestamps) > 1 and not redis_failed:
                time_since_last = all_timestamps[-1] - all_timestamps[-2]
                diffs = np.diff(all_timestamps)
                avg_time_diff = float(np.mean(diffs[-10:]))
            tx_count_10 = float(min(len(all_amounts), 10)) if not redis_failed else 0.0
    else:
        if len(all_timestamps) > 1 and not redis_failed:
            time_since_last = all_timestamps[-1] - all_timestamps[-2]
        if len(all_timestamps) > 1 and not redis_failed:
            diffs = np.diff(all_timestamps)
            avg_time_diff = float(np.mean(diffs[-10:]))
        tx_count_10 = float(min(len(all_amounts), 10)) if not redis_failed else 0.0

    # Fill final_row based on expected features
    for col in expected_features:
        # Check if Redis failed and it is a velocity/rolling feature
        if redis_failed and (col.startswith("amt_mean_") or col.startswith("amt_min_") or col.startswith("amt_max_")):
            final_row[col] = 88.34961925093133
        elif redis_failed and (col.startswith("amt_std_") or col.startswith("amt_deviation_") or col.startswith("amt_zscore_")):
            final_row[col] = 0.0
        elif redis_failed and col == "time_since_last":
            final_row[col] = 0.0
        elif redis_failed and col == "tx_count_10":
            final_row[col] = 0.0
        elif redis_failed and col == "avg_time_diff":
            final_row[col] = 0.0
        elif redis_failed and col == "amt_cumsum":
            final_row[col] = amount
        elif redis_failed and col == "amt_cumcount":
            final_row[col] = 1.0
        elif redis_failed and col == "amt_cummean":
            final_row[col] = amount
        
        # Check if it is a scaled PCA or Amount feature
        elif col in scaled_feats:
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
            if features_hash:
                hist_w = w - 1
                agg = hash_aggregates.get(hist_w, {"sum": 0.0, "count": 0})
                final_row[col] = float((agg["sum"] + amount) / (agg["count"] + 1))
            else:
                final_row[col] = float(np.mean(all_amounts[-w:]))
        elif col.startswith("amt_std_"):
            w = int(col.split("_")[-1])
            if features_hash:
                hist_w = w - 1
                agg = hash_aggregates.get(hist_w, {"sum": 0.0, "sum_sq": 0.0, "count": 0})
                mean_val = (agg["sum"] + amount) / (agg["count"] + 1)
                variance = ((agg["sum_sq"] + amount**2) / (agg["count"] + 1)) - mean_val**2
                final_row[col] = float(math.sqrt(max(variance, 0.0))) if (agg["count"] + 1) > 1 else 0.0
            else:
                final_row[col] = float(np.std(all_amounts[-w:])) if len(all_amounts[-w:]) > 1 else 0.0
        elif col.startswith("amt_min_"):
            w = int(col.split("_")[-1])
            if features_hash:
                hist_w = w - 1
                agg = hash_aggregates.get(hist_w, {"min": 0.0, "count": 0})
                final_row[col] = float(min(agg["min"], amount)) if agg["count"] > 0 else float(amount)
            else:
                final_row[col] = float(np.min(all_amounts[-w:]))
        elif col.startswith("amt_max_"):
            w = int(col.split("_")[-1])
            if features_hash:
                hist_w = w - 1
                agg = hash_aggregates.get(hist_w, {"max": 0.0, "count": 0})
                final_row[col] = float(max(agg["max"], amount)) if agg["count"] > 0 else float(amount)
            else:
                final_row[col] = float(np.max(all_amounts[-w:]))
        elif col.startswith("amt_deviation_"):
            w = int(col.split("_")[-1])
            if features_hash:
                hist_w = w - 1
                agg = hash_aggregates.get(hist_w, {"sum": 0.0, "count": 0})
                mean_val = (agg["sum"] + amount) / (agg["count"] + 1)
                final_row[col] = amount - mean_val
            else:
                mean_val = np.mean(all_amounts[-w:])
                final_row[col] = amount - mean_val
        elif col.startswith("amt_zscore_"):
            w = int(col.split("_")[-1])
            if features_hash:
                hist_w = w - 1
                agg = hash_aggregates.get(hist_w, {"sum": 0.0, "sum_sq": 0.0, "count": 0})
                mean_val = (agg["sum"] + amount) / (agg["count"] + 1)
                variance = ((agg["sum_sq"] + amount**2) / (agg["count"] + 1)) - mean_val**2
                std_val = math.sqrt(max(variance, 0.0)) if (agg["count"] + 1) > 1 else 0.0
                final_row[col] = (amount - mean_val) / std_val if std_val > 0.0 else 0.0
            else:
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
            if features_hash:
                agg = hash_aggregates.get(9, {"sum": 0.0})
                final_row[col] = float(agg["sum"] + amount)
            else:
                final_row[col] = float(np.sum(all_amounts))
        elif col == "amt_cumcount":
            if features_hash:
                agg = hash_aggregates.get(9, {"count": 0})
                final_row[col] = float(agg["count"] + 1)
            else:
                final_row[col] = float(len(all_amounts))
        elif col == "amt_cummean":
            if features_hash:
                agg = hash_aggregates.get(9, {"sum": 0.0, "count": 0})
                final_row[col] = float((agg["sum"] + amount) / (agg["count"] + 1))
            else:
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
            if features_hash:
                agg = hash_aggregates.get(2, {"sum": 0.0, "count": 0})
                mean_3 = (agg["sum"] + amount) / (agg["count"] + 1)
            else:
                mean_3 = np.mean(all_amounts[-3:])
            final_row[col] = amount / (mean_3 + 1e-8)
        else:
            final_row[col] = 0.0

    # 5. Handle KMeans behavior clustering in real-time if clusterer is loaded
    cluster_id = 0
    if behavior_clusterer is not None and behavior_clusterer_config is not None:
        try:
            cluster_features = behavior_clusterer_config["feature_names"]
            cluster_row = {}
            for col in cluster_features:
                cluster_row[col] = final_row.get(col, 0.0)
            
            cluster_df = pd.DataFrame([cluster_row], columns=cluster_features)
            cluster_id = int(behavior_clusterer.predict(cluster_df)[0])
        except Exception as e:
            import logging
            logger = logging.getLogger("api.predict")
            logger.warning(f"Failed to predict behavior cluster in real-time: {e}")
            cluster_id = 0

    # Add the cluster one-hot features if expected by the model
    for col in expected_features:
        if col == "cluster_0":
            final_row[col] = 1.0 if cluster_id == 0 else 0.0
        elif col == "cluster_1":
            final_row[col] = 1.0 if cluster_id == 1 else 0.0
        elif col == "cluster_2":
            final_row[col] = 1.0 if cluster_id == 2 else 0.0

    return pd.DataFrame([final_row], columns=expected_features)


async def safe_push_card_amount(card_id: str, amount: float, timestamp: float) -> None:
    try:
        redis = await get_redis()
        await push_card_amount(redis, card_id, amount, timestamp)
    except Exception as e:
        import logging
        logger = logging.getLogger("api.predict")
        logger.warning(f"Failed to push card amount to Redis: {e}")


async def safe_enqueue_prediction_log(
    card_id: str,
    amount: float,
    fraud_probability: float,
    is_flagged: bool,
    threshold_used: float,
    latency_ms: float,
    prediction_id: str,
) -> None:
    try:
        await enqueue_prediction_log(
            card_id=card_id,
            amount=amount,
            fraud_probability=fraud_probability,
            is_flagged=is_flagged,
            threshold_used=threshold_used,
            latency_ms=latency_ms,
            prediction_id=prediction_id,
        )
    except Exception as e:
        import logging
        logger = logging.getLogger("api.predict")
        logger.error(f"Failed to enqueue prediction log to Redis WAL: {e}")
        # Try direct write fallback to PostgreSQL
        try:
            from workers.dlq_worker import write_prediction
            from datetime import datetime, timezone
            payload = {
                "id": prediction_id,
                "card_id": card_id,
                "amount": amount,
                "fraud_probability": fraud_probability,
                "is_flagged": is_flagged,
                "threshold_used": threshold_used,
                "latency_ms": latency_ms,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await write_prediction(payload)
            logger.info("Successfully wrote prediction log directly to PostgreSQL (fallback)")
        except Exception as pg_err:
            logger.critical(f"Failed to write prediction log directly to PostgreSQL fallback: {pg_err}")


async def safe_publish_transaction(
    prediction_id: str,
    card_id: str,
    amount: float,
    fraud_probability: float,
    is_flagged: bool,
    latency_ms: float,
) -> None:
    try:
        from app.services.stream_publisher import publish_transaction
        await publish_transaction(
            prediction_id=prediction_id,
            card_id=card_id,
            amount=amount,
            fraud_probability=fraud_probability,
            is_flagged=is_flagged,
            latency_ms=latency_ms,
        )
    except Exception as e:
        import logging
        logger = logging.getLogger("api.predict")
        logger.warning(f"Failed to publish transaction event: {e}")


async def safe_publish_alert(
    prediction_id: str,
    card_id: str,
    amount: float,
    fraud_probability: float,
) -> None:
    try:
        from app.services.stream_publisher import publish_alert
        await publish_alert(
            prediction_id=prediction_id,
            card_id=card_id,
            amount=amount,
            fraud_probability=fraud_probability,
        )
    except Exception as e:
        import logging
        logger = logging.getLogger("api.predict")
        logger.warning(f"Failed to publish high-confidence fraud alert: {e}")


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
    behavior_clusterer = getattr(request.app.state, "behavior_clusterer", None)
    behavior_clusterer_config = getattr(request.app.state, "behavior_clusterer_config", None)

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

    # Fetch card history from Redis with fail-open resiliency
    redis_failed = False
    history_items = []
    features_hash = None
    try:
        redis = await get_redis()
        from app.services.redis_cache import get_card_features
        features_hash = await get_card_features(redis, body.card_id)
        if not features_hash:
            history_items = await get_card_history_with_timestamps(redis, body.card_id)
    except Exception as e:
        import logging
        logger = logging.getLogger("api.predict")
        logger.warning(f"Redis lookup failed for card {body.card_id}, falling back to population statistics. Error: {e}")
        redis_failed = True

    # Reconstruct exact features
    features_df = prepare_prediction_features(
        body.amount,
        body.hour,
        history_items,
        scaler,
        expected_features,
        behavior_clusterer=behavior_clusterer,
        behavior_clusterer_config=behavior_clusterer_config,
        redis_failed=redis_failed,
        features_hash=features_hash,
    )

    # Execute inference
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_df)
        prob = float(probs[0, 1])
    else:
        if hasattr(model, "best_iteration"):
            raw_preds = model.predict(features_df, num_iteration=model.best_iteration)
        else:
            raw_preds = model.predict(features_df)

        is_focal_loss = getattr(request.app.state, "is_focal_loss", False)
        init_score_val = getattr(request.app.state, "init_score", 0.0)

        if is_focal_loss:
            prob = float(special.expit(raw_preds[0] + init_score_val))
        else:
            prob = float(raw_preds[0])

    # Fetch dynamic decision threshold from DB (C-4)
    threshold = await config_service.get_float("shap_trigger_threshold", default=getattr(request.app.state, "threshold", 0.50))
    is_flagged = prob >= threshold
    latency_ms = (time.perf_counter() - t0) * 1000

    # Offload card amount caching and prediction WAL logging safely
    background_tasks.add_task(safe_push_card_amount, body.card_id, body.amount, time.time())

    prediction_id = str(uuid.uuid4())

    background_tasks.add_task(
        safe_enqueue_prediction_log,
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
    background_tasks.add_task(
        safe_publish_transaction,
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
            safe_publish_alert,
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
