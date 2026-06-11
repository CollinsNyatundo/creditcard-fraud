import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, cast
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import asyncio
from sqlalchemy import text
import __main__

# Add project root to path to resolve app imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.engine import AsyncSessionLocal
import model.src.calibrate_probabilities

setattr(__main__, "IsotonicCalibratedBooster", model.src.calibrate_probabilities.IsotonicCalibratedBooster)

logger = logging.getLogger(__name__)
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"



def load_model_and_data() -> Tuple[Any, pd.DataFrame, np.ndarray]:
    """Load model (calibrated if available, else baseline) and validation dataset."""
    try:
        model = joblib.load(MODELS_DIR / "calibrated_model.pkl")
        logger.info("Loaded calibrated model wrapper from calibrated_model.pkl")
    except Exception as exc:
        logger.warning("Failed to load calibrated_model.pkl: %s. Falling back to baseline.", exc)
        try:
            model = joblib.load(MODELS_DIR / "baseline_lightgbm.pkl")
            logger.info("Loaded baseline model from baseline_lightgbm.pkl")
        except Exception as baseline_exc:
            raise RuntimeError(f"Failed to load baseline model: {baseline_exc}")

    val_path = DATA_PROCESSED_DIR / "val_enhanced.csv" if (DATA_PROCESSED_DIR / "val_enhanced.csv").exists() else DATA_PROCESSED_DIR / "val.csv"
    logger.info("Loading validation data from: %s", val_path)
    try:
        val_df = pd.read_csv(val_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Validation data not found at {val_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load validation data: {exc}")

    if "Class" not in val_df.columns:
        raise KeyError("Expected 'Class' column in validation dataset")

    X_val = val_df.drop("Class", axis=1)
    y_val = cast(np.ndarray, val_df["Class"].values)
    return model, X_val, y_val


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Align validation features to model signature."""
    features = features.copy()
    
    feature_list_path = MODELS_DIR / "feature_list.json"
    if feature_list_path.exists():
        with feature_list_path.open("r") as file_handle:
            expected_features = json.load(file_handle)
    else:
        feature_names_path = MODELS_DIR / "feature_names.json"
        with feature_names_path.open("r") as file_handle:
            expected_features = json.load(file_handle)["feature_names"]

    if "Amount_Bin" in features.columns and any(col.startswith("Amount_Bin_") for col in expected_features):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        amount_bin_encoded = encoder.fit_transform(features[["Amount_Bin"]])
        encoded_names = encoder.get_feature_names_out(["Amount_Bin"])
        amount_bin_df = pd.DataFrame(
            amount_bin_encoded, columns=encoded_names, index=features.index
        )
        columns_to_drop = ["Amount_Bin"] + [
            column for column in features.columns if column in {"Amount_Low", "Amount_Medium", "Amount_High"}
        ]
        features = features.drop(columns=columns_to_drop, errors="ignore")
        features = pd.concat([features, amount_bin_df], axis=1)

    for column in expected_features:
        if column not in features.columns:
            features[column] = 0.0

    return features[expected_features]


def calculate_cost(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray) -> float:
    """Calculate total financial cost based on business utility parameters.

    FN Cost = Transaction Amount + $15 chargeback fee
    FP Cost = $50 flat customer churn/support cost
    """
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    
    fn_cost = np.sum(amounts[fn_mask] + 15.0)
    fp_cost = np.sum(fp_mask) * 50.0
    return float(fn_cost + fp_cost)


def search_dual_thresholds(model, X: pd.DataFrame, y: np.ndarray, amounts: np.ndarray) -> Tuple[float, float, Dict[str, Any], np.ndarray, np.ndarray]:
    """Search for cost_min and recall_target thresholds using 0.01 steps."""
    # Obtain probabilities from model or calibrated wrapper
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        if hasattr(model, "best_iteration"):
            probabilities = model.predict(X, num_iteration=model.best_iteration)
        else:
            probabilities = model.predict(X)

    thresholds = np.arange(0.01, 1.0, 0.01)
    precision_values = []
    recall_values = []
    f1_values = []
    g_mean_values = []
    cost_values = []

    y_list = y.tolist()

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        pred_list = predictions.tolist()
        
        # Calculate metrics
        precision = precision_score(y_list, pred_list, zero_division=0)
        recall = recall_score(y_list, pred_list, zero_division=0)
        f1 = f1_score(y_list, pred_list, zero_division=0)
        
        cm = confusion_matrix(y_list, pred_list)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        g_mean = np.sqrt(sensitivity * specificity)
        
        cost = calculate_cost(y, predictions, amounts)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        g_mean_values.append(g_mean)
        cost_values.append(cost)

    cost_values = np.array(cost_values)
    
    # 1. Cost minimizing threshold
    best_cost_index = int(np.argmin(cost_values))
    cost_min = float(thresholds[best_cost_index])

    # 2. Recall constrained threshold (Recall >= 0.85)
    qualified_indices = [idx for idx, rec in enumerate(recall_values) if rec >= 0.85]
    if qualified_indices:
        # Pick the highest threshold that satisfies the Recall target to minimize false positives
        best_recall_index = int(max(qualified_indices))
        recall_target = float(thresholds[best_recall_index])
    else:
        # Fallback to threshold maximizing recall
        best_recall_index = int(np.argmax(recall_values))
        recall_target = float(thresholds[best_recall_index])
        logger.warning("No threshold satisfied Recall >= 0.85 gate on validation set. Using best effort.")

    metrics_cost_min = {
        "precision": float(precision_values[best_cost_index]),
        "recall": float(recall_values[best_cost_index]),
        "f1_score": float(f1_values[best_cost_index]),
        "g_mean": float(g_mean_values[best_cost_index]),
        "cost": float(cost_values[best_cost_index])
    }

    metrics_recall_target = {
        "precision": float(precision_values[best_recall_index]),
        "recall": float(recall_values[best_recall_index]),
        "f1_score": float(f1_values[best_recall_index]),
        "g_mean": float(g_mean_values[best_recall_index]),
        "cost": float(cost_values[best_recall_index])
    }

    results = {
        "cost_min": cost_min,
        "recall_target": recall_target,
        "metrics_cost_min": metrics_cost_min,
        "metrics_recall_target": metrics_recall_target
    }

    return cost_min, recall_target, results, thresholds, cost_values


def plot_cost_curve(thresholds: np.ndarray, cost_values: np.ndarray, cost_min: float, recall_target: float):
    """Plot and save the Cost vs Threshold curve."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, cost_values, label="Total Financial Cost ($)", color="#1f77b4", linewidth=2.5)
    plt.axvline(x=cost_min, color="#d62728", linestyle="--", linewidth=1.5, label=f"Cost Min Threshold ({cost_min:.2f})")
    plt.axvline(x=recall_target, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Recall Target Threshold ({recall_target:.2f})")
    plt.title("Total Business Cost vs. Decision Threshold", fontsize=14, fontweight="bold")
    plt.xlabel("Decision Threshold", fontsize=12)
    plt.ylabel("Total Financial Cost ($)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    
    plot_path = REPORTS_DIR / "cost_vs_threshold_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Cost vs. Threshold curve to %s", plot_path)


async def sync_threshold_to_db(threshold_value: float):
    """Sync the active threshold to the system_config database table."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("""
                    INSERT INTO system_config (key, value, updated_at)
                    VALUES ('shap_trigger_threshold', :value, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = :value, updated_at = NOW()
                """),
                {"value": f"{threshold_value:.4f}"}
            )
            await session.commit()
            logger.info("Successfully synced active threshold %.4f to PostgreSQL system_config", threshold_value)
    except Exception as exc:
        logger.error("Failed to sync threshold to database: %s", exc)


def save_results(cost_min: float, recall_target: float, metrics: Dict[str, Any]):
    """Save threshold configs to optimal_threshold_v2.json, preserving focal loss details."""
    config_v2_path = MODELS_DIR / "optimal_threshold_v2.json"
    
    # Defaults / existing config load
    init_score = 0.0
    is_focal_loss = False
    focal_loss_params = {}
    active_selection = "recall_target"  # Default active key

    if config_v2_path.exists():
        try:
            with config_v2_path.open("r") as f:
                existing_config = json.load(f)
                init_score = existing_config.get("init_score", 0.0)
                is_focal_loss = existing_config.get("is_focal_loss", False)
                active_selection = existing_config.get("active", "recall_target")
                focal_loss_params = {
                    "init_score": init_score,
                    "is_focal_loss": is_focal_loss,
                    "alpha": existing_config.get("alpha"),
                    "gamma": existing_config.get("gamma"),
                }
        except Exception as exc:
            logger.warning("Could not load existing optimal_threshold_v2.json: %s", exc)

    resolved_threshold = cost_min if active_selection == "cost_min" else recall_target

    payload = {
        "cost_min": cost_min,
        "recall_target": recall_target,
        "active": active_selection,
        "threshold": resolved_threshold,
        **focal_loss_params
    }

    with config_v2_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved optimal threshold configs to %s", config_v2_path)
    
    # Save detailed metrics to reports
    reports_path = REPORTS_DIR / "threshold_pareto_results.json"
    with reports_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved pareto results report to %s", reports_path)

    # Database sync
    asyncio.run(sync_threshold_to_db(resolved_threshold))


def main():
    """Main execution entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting dual threshold optimization...")
    
    model, X_val, y_val = load_model_and_data()
    amounts = cast(np.ndarray, X_val["Amount"].values)
    
    prepared = prepare_features(X_val)
    cost_min, recall_target, results, thresholds, cost_values = search_dual_thresholds(
        model, prepared, y_val, amounts
    )
    
    logger.info("Optimized Cost-Min Threshold: %.4f (Cost: $%.2f)", cost_min, results["metrics_cost_min"]["cost"])
    logger.info("Recall-Constrained Threshold (>=0.85): %.4f (Recall: %.4f)", recall_target, results["metrics_recall_target"]["recall"])
    
    plot_cost_curve(thresholds, cost_values, cost_min, recall_target)
    save_results(cost_min, recall_target, results)


if __name__ == "__main__":
    main()
