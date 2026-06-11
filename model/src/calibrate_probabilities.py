import logging
import os
import sys
import json
from pathlib import Path
from typing import Tuple, cast

import pandas as pd

import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import lightgbm as lgb
from scipy.special import expit

# Add project root to sys.path first to ensure absolute imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class IsotonicCalibratedBooster:
    """LightGBM Booster with logistic regression post-hoc probability calibration (Platt Scaling).

    Keeps the class name for backward compatibility with pickling and API endpoints.
    """

    def __init__(self, booster: lgb.Booster, threshold: float = 0.5, is_focal_loss: bool = False, init_score: float = 0.0) -> None:
        self.booster = booster
        self.threshold = threshold
        self.is_focal_loss = is_focal_loss
        self.init_score = init_score
        self.classes_ = np.array([0, 1])
        self._calibrator: LogisticRegression | None = None

    def _raw_proba(self, X) -> np.ndarray:
        raw = np.asarray(
            self.booster.predict(X, num_iteration=self.booster.best_iteration)
        ).reshape(-1)
        if self.is_focal_loss:
            return expit(raw + self.init_score)
        return raw

    def fit(self, X_cal, y_cal) -> "IsotonicCalibratedBooster":
        """Fit logistic regression calibrator on a hold-out calibration set."""
        raw = self._raw_proba(X_cal)
        self._calibrator = LogisticRegression()
        self._calibrator.fit(pd.DataFrame(raw.reshape(-1, 1)), y_cal)
        logger.info("Logistic regression calibrator fitted on %d samples.", len(y_cal))
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self._raw_proba(X)
        if self._calibrator is not None:
            cal = self._calibrator.predict_proba(pd.DataFrame(raw.reshape(-1, 1)))[:, 1]
        else:
            cal = raw
        out = np.empty((len(cal), 2), dtype=np.float64)
        out[:, 0] = 1.0 - cal
        out[:, 1] = cal
        return out


def load_baseline_model(model_dir: Path = MODELS_DIR) -> Tuple[lgb.Booster, float, bool, float]:
    """Load the best available model and optimal threshold configs."""
    logger.info("Loading model and configs...")
    
    if (model_dir / "optimized_lightgbm.pkl").exists():
        model = joblib.load(model_dir / "optimized_lightgbm.pkl")
        logger.info("Loaded optimized flagship model from optimized_lightgbm.pkl")
    else:
        model = joblib.load(model_dir / "baseline_lightgbm.pkl")
        logger.info("Loaded baseline model from baseline_lightgbm.pkl")

    threshold_path = model_dir / "optimal_threshold_v2.json"
    if not threshold_path.exists():
        threshold_path = model_dir / "optimal_threshold.json"

    optimal_threshold = 0.5
    is_focal_loss = False
    init_score = 0.0

    with threshold_path.open("r") as file_handler:
        config_data = json.load(file_handler)
        is_focal_loss = config_data.get("is_focal_loss", False)
        init_score = config_data.get("init_score", 0.0)
        
        # Check active threshold key if present
        active_key = config_data.get("active", "recall_target")
        if active_key in config_data:
            optimal_threshold = float(config_data[active_key])
        else:
            optimal_threshold = float(config_data.get("threshold", 0.5))

    logger.info(
        "Model configs: threshold=%.4f, is_focal_loss=%s, init_score=%.4f",
        optimal_threshold, is_focal_loss, init_score
    )
    return model, optimal_threshold, is_focal_loss, init_score


def load_datasets(
    data_dir: Path = DATA_PROCESSED_DIR,
) -> Tuple[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray]]:
    """Load validation and test datasets, choosing enhanced versions if available."""
    logger.info("Loading validation and test data...")
    val_path = data_dir / "val_enhanced.csv" if (data_dir / "val_enhanced.csv").exists() else data_dir / "val.csv"
    test_path = data_dir / "test_enhanced.csv" if (data_dir / "test_enhanced.csv").exists() else data_dir / "test.csv"

    logger.info("Using val data: %s", val_path)
    logger.info("Using test data: %s", test_path)

    validation_dataframe = pd.read_csv(val_path)
    test_dataframe = pd.read_csv(test_path)

    def _split(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        if "Class" not in dataframe.columns:
            raise KeyError("Expected 'Class' column in dataset")
        features = dataframe.drop("Class", axis=1)
        target = cast(np.ndarray, dataframe["Class"].values)
        return features, target

    validation_features, validation_target = _split(validation_dataframe)
    test_features, test_target = _split(test_dataframe)
    logger.info(
        "Validation set shape: %s, Test set shape: %s",
        validation_features.shape,
        test_features.shape,
    )
    return (validation_features, validation_target), (test_features, test_target)


def prepare_features(
    features: pd.DataFrame, models_dir: Path = MODELS_DIR
) -> pd.DataFrame:
    """Align feature set with the model's expected signature."""
    aligned_features = features.copy()
    
    # Load expected features
    feature_list_path = models_dir / "feature_list.json"
    if feature_list_path.exists():
        with feature_list_path.open("r") as file_handler:
            expected_features = json.load(file_handler)
    else:
        feature_names_path = models_dir / "feature_names.json"
        with feature_names_path.open("r") as file_handler:
            expected_features = json.load(file_handler)["feature_names"]

    # If dataset has Amount_Bin but model expects one-hot columns (baseline fallback)
    if "Amount_Bin" in aligned_features.columns and any(col.startswith("Amount_Bin_") for col in expected_features):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        amount_bin_encoded = encoder.fit_transform(aligned_features[["Amount_Bin"]])
        encoded_feature_names = encoder.get_feature_names_out(["Amount_Bin"])
        amount_bin_dataframe = pd.DataFrame(
            amount_bin_encoded, columns=encoded_feature_names, index=aligned_features.index
        )
        columns_to_drop = ["Amount_Bin"] + [
            column
            for column in aligned_features.columns
            if column in {"Amount_Low", "Amount_Medium", "Amount_High"}
        ]
        aligned_features = aligned_features.drop(columns=columns_to_drop, errors="ignore")
        aligned_features = pd.concat([aligned_features, amount_bin_dataframe], axis=1)

    for column in expected_features:
        if column not in aligned_features.columns:
            aligned_features[column] = 0.0

    return aligned_features[expected_features]


def compute_metrics(
    ground_truth: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Compute classification and ranking metrics."""
    predicted_classes = (predicted_labels >= threshold).astype(int)
    
    # Convert numpy arrays to standard Python lists to avoid type stub conflicts
    y_true_list = ground_truth.tolist()
    y_pred_list = predicted_classes.tolist()
    y_prob_list = predicted_probabilities.tolist()
    
    cm = confusion_matrix(y_true_list, y_pred_list)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = float(np.sqrt(sensitivity * specificity))
    
    return {
        "precision": float(precision_score(y_true_list, y_pred_list, zero_division=0)),
        "recall": float(recall_score(y_true_list, y_pred_list, zero_division=0)),
        "f1_score": float(f1_score(y_true_list, y_pred_list, zero_division=0)),
        "g_mean": g_mean,
        "pr_auc": float(average_precision_score(y_true_list, y_prob_list)),
        "auprc": float(average_precision_score(y_true_list, y_prob_list)),
        "confusion_matrix": cm.tolist(),
    }


def print_summary(before_metrics: dict, after_metrics: dict) -> None:
    """Print a concise before/after summary with deltas."""
    print("\n=== Calibration Results Summary ===")
    print(f"{'Metric':<12} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 46)
    for metric in ["precision", "recall", "f1_score", "g_mean", "pr_auc"]:
        before_value = before_metrics[metric]
        after_value = after_metrics[metric]
        delta_value = after_value - before_value
        sign = "+" if delta_value >= 0 else ""
        print(f"{metric:<12} {before_value:>10.4f} {after_value:>10.4f} {sign}{delta_value:>9.4f}")
    print("=== End of Summary ===")


def main() -> None:
    """Calibrate model probabilities using isotonic regression."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, threshold, is_focal_loss, init_score = load_baseline_model()
    (validation_features, validation_target), (test_features, test_target) = load_datasets()

    validation_prepared = prepare_features(validation_features)
    test_prepared = prepare_features(test_features)

    calibrated_booster = IsotonicCalibratedBooster(
        model, threshold=threshold, is_focal_loss=is_focal_loss, init_score=init_score
    )

    logger.info("Fitting isotonic calibrator on validation set (%d samples)...", len(validation_target))
    calibrated_booster.fit(validation_prepared, validation_target)
    logger.info("Calibration complete.")

    # Before: raw booster probabilities on test set
    if is_focal_loss:
        raw_preds_test = np.asarray(
            model.predict(test_prepared, num_iteration=model.best_iteration)
        ).reshape(-1)
        raw_proba_test = expit(raw_preds_test + init_score)
    else:
        raw_proba_test = np.asarray(
            model.predict(test_prepared, num_iteration=model.best_iteration)
        ).reshape(-1)
        
    before_metrics = compute_metrics(
        test_target, raw_proba_test, raw_proba_test, threshold
    )

    # After: calibrated probabilities on test set
    predicted_probabilities_after = calibrated_booster.predict_proba(test_prepared)[:, 1]
    after_metrics = compute_metrics(
        test_target, predicted_probabilities_after, predicted_probabilities_after, threshold
    )

    save_path = MODELS_DIR / "calibrated_model.pkl"
    logger.info("Saving calibrated model wrapper to %s", save_path)
    joblib.dump(calibrated_booster, save_path)

    # AUPRC regression gate check
    if after_metrics["auprc"] < 0.70:
        logger.error("ERROR: AUPRC regression detected (calibrated AUPRC: %.4f < 0.70)", after_metrics["auprc"])

    comparison = {
        "before": before_metrics,
        "after": after_metrics,
        "threshold": threshold,
        "calibration_method": "isotonic",
    }
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving metrics comparison to %s", REPORTS_DIR / "calibration_results.json")
    with (REPORTS_DIR / "calibration_results.json").open("w") as file_handler:
        json.dump(comparison, file_handler, indent=2)

    print_summary(before_metrics, after_metrics)
    logger.info("Calibration pipeline completed.")


if __name__ == "__main__":
    main()
