"""
Final Model Evaluation — Task 5 (f1-precision-improvement-plan)
================================================================
Loads the baseline LightGBM model plus optional improvements:
  - Calibrated probability wrapper (models/calibrated_model.pkl)
  - Pareto-optimized threshold (models/optimal_threshold_v2.json)

Produces a side-by-side baseline vs. optimised metrics report and saves
results to reports/final_performance_evaluation.json.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add project root to sys.path first to ensure absolute imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import model.src.calibrate_probabilities
import __main__
setattr(__main__, "IsotonicCalibratedBooster", model.src.calibrate_probabilities.IsotonicCalibratedBooster)

from model.src.feature_coverage_check import check_feature_explanations_coverage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline_model() -> Tuple[lgb.Booster, float]:
    """Load the baseline LightGBM Booster and its optimal threshold."""
    model_path = MODELS_DIR / "baseline_lightgbm.pkl"
    threshold_path = MODELS_DIR / "optimal_threshold.json"

    logger.info("Loading baseline model from %s", model_path)
    model = joblib.load(model_path)

    with threshold_path.open("r") as fh:
        baseline_threshold = json.load(fh)["threshold"]

    logger.info("Baseline threshold: %.4f", baseline_threshold)
    return model, baseline_threshold


def load_calibrated_model() -> Optional[object]:
    """Return calibrated model wrapper if it exists, else None."""
    calibrated_path = MODELS_DIR / "calibrated_model.pkl"
    if calibrated_path.exists():
        logger.info("Loading calibrated model from %s", calibrated_path)
        return joblib.load(calibrated_path)
    logger.warning("Calibrated model not found at %s — skipping calibration", calibrated_path)
    return None


def load_optimized_threshold() -> Optional[float]:
    """Return optimized threshold from optimal_threshold_v2.json based on active selection key."""
    config_path = MODELS_DIR / "optimal_threshold_v2.json"
    if config_path.exists():
        with config_path.open("r") as fh:
            data = json.load(fh)
        threshold = float(data.get("threshold", 0.5))
        logger.info(
            "Optimized threshold: %.4f resolved from pre-resolved threshold field",
            threshold,
        )
        return threshold
    logger.warning("Optimized threshold config not found at %s — using baseline threshold", config_path)
    return None


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load test split, choosing enhanced version if available."""
    test_path = DATA_DIR / "test_enhanced.csv"
    if not test_path.exists():
        test_path = DATA_DIR / "test.csv"
    logger.info("Loading test data from %s", test_path)
    df = pd.read_csv(test_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    logger.info("Test data shape: %s | Fraud rate: %.4f", df.shape, y.mean())
    return X, y


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(X: pd.DataFrame, feature_config_name: str = "feature_names.json") -> pd.DataFrame:
    """Align test features with the model's expected signature."""
    X = X.copy()
    feature_names_path = MODELS_DIR / feature_config_name
    if feature_config_name == "feature_list.json" and not feature_names_path.exists():
        feature_names_path = MODELS_DIR / "feature_names.json"

    with feature_names_path.open("r") as fh:
        data = json.load(fh)
        if isinstance(data, dict):
            expected_features = data["feature_names"]
        else:
            expected_features = data

    if "Amount_Bin" in X.columns and any(col.startswith("Amount_Bin_") for col in expected_features):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(X[["Amount_Bin"]])
        enc_cols = encoder.get_feature_names_out(["Amount_Bin"])
        enc_df = pd.DataFrame(encoded, columns=enc_cols, index=X.index)
        drop_cols = ["Amount_Bin"] + [
            c for c in X.columns if c in {"Amount_Low", "Amount_Medium", "Amount_High"}
        ]
        X = pd.concat([X.drop(columns=drop_cols, errors="ignore"), enc_df], axis=1)

    for col in expected_features:
        if col not in X.columns:
            X[col] = 0.0

    return X[expected_features]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return fraud probability array regardless of model type."""
    if hasattr(model, "predict_proba"):
        return cast(np.ndarray, model.predict_proba(X)[:, 1])
    if isinstance(model, lgb.Booster):
        return cast(np.ndarray, model.predict(X, num_iteration=model.best_iteration))
    return cast(np.ndarray, model.predict(X))


def plot_precision_recall_curve(y_true, y_proba, label: str):
    """Plot Precision-Recall curve and save to reports/."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall.tolist(), precision.tolist(), color='#1f77b4', lw=2.5, label=f'{label} (AUPRC = {average_precision_score(y_true, y_proba):.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = REPORTS_DIR / "precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Precision-Recall Curve to %s", plot_path)


def compute_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    label: str,
) -> Dict:
    """Compute and log classification metrics for a given model/threshold."""
    y_proba = _predict_proba(model, X)
    y_pred = (y_proba >= threshold).astype(int)

    y_true_list = y.tolist()
    y_pred_list = y_pred.tolist()
    y_prob_list = y_proba.tolist()

    f1 = f1_score(y_true_list, y_pred_list)
    prec = precision_score(y_true_list, y_pred_list)
    rec = recall_score(y_true_list, y_pred_list)
    roc = roc_auc_score(y_true_list, y_prob_list)
    pr_auc = average_precision_score(y_true_list, y_prob_list)
    cm = confusion_matrix(y_true_list, y_pred_list)
    report = classification_report(y_true_list, y_pred_list, output_dict=True)

    # Compute G-mean
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = float(np.sqrt(sensitivity * specificity))

    pr_auc_status = "PASS" if pr_auc > 0.70 else "FAIL"
    recall_status = "PASS" if rec >= 0.85 else "FAIL"

    logger.info("── %s (threshold=%.4f) ──────────────────────", label, threshold)
    logger.info("  Recall          : %.4f [%s] (Gate: >=0.85)", rec, recall_status)
    logger.info("  PR-AUC (AUPRC)  : %.4f [%s] (Gate: >0.70)", pr_auc, pr_auc_status)
    logger.info("  G-Mean          : %.4f", g_mean)
    logger.info("  F1-Score        : %.4f", f1)
    logger.info("  ROC-AUC         : %.4f", roc)
    logger.info("  Precision       : %.4f", prec)
    
    # Emit explicit PASS/FAIL gate lines
    logger.info("[%s] Recall ≥ 0.85: %.4f", recall_status, rec)
    logger.info("[%s] AUPRC > 0.70: %.4f", pr_auc_status, pr_auc)
    logger.info("[SECONDARY] F1 > 0.80: %.4f", f1)
    logger.info("  Confusion matrix:\n%s", cm)

    return {
        "label": label,
        "threshold": threshold,
        "pr_auc": pr_auc,
        "recall": rec,
        "f1_score": f1,
        "g_mean": g_mean,
        "precision": prec,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def benchmark_latency(model, X: pd.DataFrame, sample_size: int = 1_000) -> Dict:
    """Single-row inference latency benchmark (ms)."""
    logger.info("Benchmarking inference latency on %d samples…", sample_size)
    idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[idx.tolist()]

    # Warm-up
    _predict_proba(model, X_sample.iloc[0:1])

    latencies = []
    for i in range(len(X_sample)):
        t0 = time.perf_counter()
        _predict_proba(model, X_sample.iloc[i:i + 1])
        latencies.append((time.perf_counter() - t0) * 1_000)

    lat = np.array(latencies)
    stats = {
        "mean_ms": float(np.mean(lat)),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "max_ms": float(np.max(lat)),
    }
    logger.info(
        "  p50=%.3fms  p95=%.3fms  p99=%.3fms  max=%.3fms",
        stats["median_ms"], stats["p95_ms"], stats["p99_ms"], stats["max_ms"],
    )
    return stats


def build_comparison_report(baseline: Dict, optimised: Optional[Dict]) -> Dict:
    """Build a side-by-side delta table."""
    if optimised is None:
        return {"note": "No optimised variant available — baseline only.", "baseline": baseline}

    def delta(key: str) -> float:
        return round(optimised[key] - baseline[key], 6)

    return {
        "baseline": baseline,
        "optimised": optimised,
        "deltas": {
            "pr_auc": delta("pr_auc"),
            "recall": delta("recall"),
            "f1_score": delta("f1_score"),
            "g_mean": delta("g_mean"),
            "precision": delta("precision"),
            "roc_auc": delta("roc_auc"),
        },
    }


def deployment_readiness(metrics: Dict, latency: Dict) -> Dict:
    """Check target metric gates for production deployment.

    - Recall >= 0.85
    - AUPRC > 0.70
    - Latency p95 < 10ms
    """
    recall_ok = metrics["recall"] >= 0.85
    auprc_ok = metrics["pr_auc"] > 0.70
    lat_ok = latency["p95_ms"] < 10.0
    
    logger.info("── Deployment Readiness ─────────────────────────────────")
    logger.info("  Recall >= 0.85 : %s  (%.4f)", "PASS" if recall_ok else "FAIL", metrics["recall"])
    logger.info("  AUPRC  > 0.70  : %s  (%.4f)", "PASS" if auprc_ok else "FAIL", metrics["pr_auc"])
    logger.info("  p95 < 10ms     : %s  (%.4fms)", "PASS" if lat_ok else "FAIL", latency["p95_ms"])
    logger.info("  Overall        : %s", "READY" if (recall_ok and auprc_ok and lat_ok) else "NOT READY")
    
    return {
        "recall_requirement_met": recall_ok,
        "auprc_requirement_met": auprc_ok,
        "latency_requirement_met": lat_ok,
        "overall_ready": recall_ok and auprc_ok and lat_ok,
        "recall_gap": round(0.85 - metrics["recall"], 6) if not recall_ok else 0.0,
        "auprc_gap": round(0.70 - metrics["pr_auc"], 6) if not auprc_ok else 0.0,
        "latency_gap_ms": round(latency["p95_ms"] - 10.0, 6) if not lat_ok else 0.0,
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(i) for i in obj]
    return obj


def save_results(payload: Dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "final_performance_evaluation.json"
    with out_path.open("w") as fh:
        json.dump(_serialise(payload), fh, indent=2)
    logger.info("Results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Final Model Evaluation (Task 5) ===")

    # --- Load assets ---
    baseline_model, baseline_threshold = load_baseline_model()
    calibrated_model = load_calibrated_model()
    optimized_threshold = load_optimized_threshold()

    # --- Load and prepare test data ---
    X_test_raw, y_test = load_test_data()
    X_test_baseline = prepare_features(X_test_raw, "feature_names.json")
    X_test_optimised = prepare_features(X_test_raw, "feature_list.json")

    # --- Feature coverage check ---
    logger.info("Running feature explanations coverage check…")
    check_feature_explanations_coverage(list(X_test_optimised.columns))

    # --- Baseline evaluation ---
    baseline_metrics = compute_metrics(
        baseline_model, X_test_baseline, y_test, baseline_threshold, label="Baseline"
    )

    # --- Optimised evaluation (calibrated model + optimized threshold if available) ---
    eval_model = calibrated_model if calibrated_model is not None else baseline_model
    eval_threshold = optimized_threshold if optimized_threshold is not None else baseline_threshold
    optimised_label = (
        f"{'Calibrated' if calibrated_model else 'Baseline'}"
        f"+{'OptimizedThresh' if optimized_threshold else 'BaselineThresh'}"
    )

    optimised_metrics: Optional[Dict] = None
    if calibrated_model is not None or optimized_threshold is not None:
        optimised_metrics = compute_metrics(
            eval_model, X_test_optimised, y_test, eval_threshold, label=optimised_label
        )
        
        # Plot PR Curve for the optimized model configuration
        y_proba_opt = _predict_proba(eval_model, X_test_optimised)
        plot_precision_recall_curve(y_test, y_proba_opt, label=optimised_label)
    else:
        logger.info("No optimisations available — reporting baseline only.")

    # --- Latency benchmark on eval model ---
    latency = benchmark_latency(eval_model, X_test_optimised)

    # --- Deployment readiness (against best available metrics) ---
    best_metrics = optimised_metrics if optimised_metrics is not None else baseline_metrics
    readiness = deployment_readiness(best_metrics, latency)

    # --- Comparison report ---
    comparison = build_comparison_report(baseline_metrics, optimised_metrics)

    # --- Save ---
    payload = {
        "comparison": comparison,
        "latency_benchmark": latency,
        "deployment_readiness": readiness,
    }
    save_results(payload)
    logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
