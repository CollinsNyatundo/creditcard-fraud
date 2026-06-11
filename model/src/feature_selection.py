import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_artifacts():
    """Load model and datasets needed for feature selection."""
    model = joblib.load(MODELS_DIR / "baseline_lightgbm.pkl")
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")

    if "Class" not in val_df.columns or "Class" not in test_df.columns:
        raise KeyError("Expected 'Class' column in validation/test datasets")

    X_val = val_df.drop("Class", axis=1)
    y_val = val_df["Class"].values
    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"].values

    feature_names_path = MODELS_DIR / "feature_names.json"
    with feature_names_path.open("r") as file_handle:
        expected_features = json.load(file_handle)["feature_names"]

    return model, X_val, y_val, X_test, y_test, expected_features


def extract_feature_importance(model) -> np.ndarray:
    """Return feature importance scores from the trained LightGBM model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "feature_importance"):
        importance = model.feature_importance()
    else:
        raise AttributeError("Loaded model does not expose feature importances")
    return np.asarray(importance, dtype=float)


def select_top_features(model, expected_features: List[str], keep_ratio: float):
    """Select top-k features by importance."""
    importance = extract_feature_importance(model)
    feature_importance_pairs = sorted(
        zip(expected_features, importance), key=lambda item: item[1], reverse=True
    )
    selected_count = max(1, int(len(feature_importance_pairs) * keep_ratio))
    selected_features = [feature for feature, _ in feature_importance_pairs[:selected_count]]
    return selected_features


def align_features(features: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Align dataset to selected features."""
    aligned = features.copy()
    for feature in selected_features:
        if feature not in aligned.columns:
            aligned[feature] = 0.0
    return aligned[selected_features]


def evaluate_configuration(model, X, y, selected_features):
    """Evaluate a single feature configuration."""
    prepared = align_features(X, selected_features)
    probabilities = model.predict(prepared, num_iteration=model.best_iteration)
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
    }


def save_results(results: Dict, output_path: Path):
    """Save feature selection results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file_handle:
        json.dump(results, file_handle, indent=2)
    logger.info("Saved feature selection results to %s", output_path)


def main():
    """Compare model performance across feature subsets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting feature selection analysis...")

    model, X_val, y_val, X_test, y_test, expected_features = load_artifacts()
    feature_importance = extract_feature_importance(model)

    ranked_features = sorted(
        zip(expected_features, feature_importance), key=lambda item: item[1], reverse=True
    )

    configurations = [
        ("all", 1.0),
        ("top_75", 0.75),
        ("top_50", 0.50),
    ]

    summary = {
        "feature_count_total": len(expected_features),
        "configurations": {},
    }

    logger.info("=== Feature Selection Results ===")
    logger.info(
        "%-10s %-25s %-10s %-10s %-10s %-10s",
        "Config",
        "FeatureCount",
        "Precision",
        "Recall",
        "F1",
        "Threshold",
    )

    for config_name, keep_ratio in configurations:
        selected_features = select_top_features(model, expected_features, keep_ratio)
        val_metrics = evaluate_configuration(model, X_val, y_val, selected_features)
        test_metrics = evaluate_configuration(model, X_test, y_test, selected_features)

        summary["configurations"][config_name] = {
            "feature_count": len(selected_features),
            "keep_ratio": keep_ratio,
            "validation": val_metrics,
            "test": test_metrics,
            "top_features": selected_features[:10],
        }

        logger.info(
            "%-10s %-25s %-10.4f %-10.4f %-10.4f %-10.4f",
            config_name,
            str(len(selected_features)),
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1_score"],
            0.5,
        )

    save_results(summary, REPORTS_DIR / "feature_selection_results.json")
    logger.info("Feature selection analysis complete.")


if __name__ == "__main__":
    main()
