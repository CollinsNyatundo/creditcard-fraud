import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from scipy.special import expit

def main():
    print("=" * 60)
    print("BOOTSTRAP STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Load model and configurations
    model_path = "./models/optimized_lightgbm.pkl"
    calibrated_path = "./models/calibrated_model.pkl"
    feature_list_path = "./models/feature_list.json"
    threshold_path = "./models/optimal_threshold_v2.json"
    
    if not os.path.exists(model_path) or not os.path.exists(feature_list_path) or not os.path.exists(threshold_path):
        print("[FAIL] Optimized model or configs not found. Run training first.")
        return
        
    # Ensure IsotonicCalibratedBooster is registered in __main__
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from model.src.calibrate_probabilities import IsotonicCalibratedBooster
        import __main__
        __main__.IsotonicCalibratedBooster = IsotonicCalibratedBooster
    except ImportError:
        pass

    with open(feature_list_path, 'r') as f:
        feature_cols = json.load(f)
    with open(threshold_path, 'r') as f:
        config_data = json.load(f)
        threshold = config_data.get('threshold', 0.5)
        init_score = config_data.get('init_score', 0.0)
        is_focal_loss = config_data.get('is_focal_loss', False)
        
    # Load test data
    test_path = "./data/processed/test_enhanced.csv"
    if not os.path.exists(test_path):
        print(f"[FAIL] Test data not found at {test_path}")
        return
        
    df_test = pd.read_csv(test_path)
    X_test = df_test[feature_cols]
    y_test = df_test['Class'].values
    
    # Load model and predict probabilities
    if os.path.exists(calibrated_path):
        print("Loading calibrated model wrapper...")
        model = joblib.load(calibrated_path)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        print("Loading raw optimized model...")
        model = joblib.load(model_path)
        raw_preds = model.predict(X_test)
        if is_focal_loss:
            y_proba = expit(raw_preds + init_score)
        else:
            y_proba = raw_preds
        
    y_pred = (y_proba >= threshold).astype(int)
    
    # Original metrics
    orig_recall = recall_score(y_test, y_pred)
    orig_auprc = average_precision_score(y_test, y_proba)
    orig_f1 = f1_score(y_test, y_pred)
    orig_precision = precision_score(y_test, y_pred)
    orig_roc_auc = roc_auc_score(y_test, y_proba)

    recall_status = "PASS" if orig_recall >= 0.85 else "FAIL"
    auprc_status = "PASS" if orig_auprc > 0.70 else "FAIL"

    print(f"Original Recall   : {orig_recall:.4f} [{recall_status}] (Gate: >=0.85 — PRIMARY)")
    print(f"Original AUPRC    : {orig_auprc:.4f} [{auprc_status}] (Gate: >0.70 — RANKING)")
    print(f"Original F1       : {orig_f1:.4f} (secondary)")
    print(f"Original Precision: {orig_precision:.4f}")
    print(f"Original ROC AUC  : {orig_roc_auc:.4f} (secondary, backward compat.)")
    
    # Run Bootstrap — primary statistic is Recall (gate: >= 0.85)
    print("Running bootstrap resampling (B = 10,000)...")
    np.random.seed(42)
    B = 10000
    bootstrap_recall_scores = []
    bootstrap_f1_scores = []

    n_samples = len(y_test)

    for i in range(B):
        # Sample indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_b = y_test[indices]
        y_proba_b = y_proba[indices]
        y_pred_b = (y_proba_b >= threshold).astype(int)

        bootstrap_recall_scores.append(recall_score(y_true_b, y_pred_b, zero_division=0))
        bootstrap_f1_scores.append(f1_score(y_true_b, y_pred_b, zero_division=0))

    bootstrap_recall_scores = np.array(bootstrap_recall_scores)
    bootstrap_f1_scores = np.array(bootstrap_f1_scores)

    mean_recall = float(np.mean(bootstrap_recall_scores))
    median_recall = float(np.median(bootstrap_recall_scores))
    ci_95_recall = [
        float(np.percentile(bootstrap_recall_scores, 2.5)),
        float(np.percentile(bootstrap_recall_scores, 97.5)),
    ]

    # P-value for null hypothesis: Recall < 0.85
    p_value_recall = float(np.mean(bootstrap_recall_scores < 0.85))

    # F1 bootstrap (secondary metric)
    mean_f1 = float(np.mean(bootstrap_f1_scores))
    ci_95_f1 = [
        float(np.percentile(bootstrap_f1_scores, 2.5)),
        float(np.percentile(bootstrap_f1_scores, 97.5)),
    ]

    print("\n--- Bootstrap Results (PRIMARY: Recall) ---")
    print(f"Bootstrap Mean Recall : {mean_recall:.4f}")
    print(f"Bootstrap Median Recall: {median_recall:.4f}")
    print(f"95% CI Recall         : [{ci_95_recall[0]:.4f}, {ci_95_recall[1]:.4f}]")
    print(f"p-value (Recall < 0.85): {p_value_recall:.4f}")
    print(f"\nBootstrap Mean F1 (secondary): {mean_f1:.4f}")
    print(f"95% CI F1             : [{ci_95_f1[0]:.4f}, {ci_95_f1[1]:.4f}]")
    
    results = {
        "original_recall": orig_recall,
        "original_auprc": orig_auprc,
        "original_f1": orig_f1,
        "original_precision": orig_precision,
        "original_roc_auc": orig_roc_auc,
        "recall_gate": {"target": 0.85, "status": recall_status, "value": orig_recall},
        "auprc_gate": {"target": 0.70, "status": auprc_status, "value": orig_auprc},
        "bootstrap_recall": {
            "mean": mean_recall,
            "median": median_recall,
            "ci_95": ci_95_recall,
        },
        "bootstrap_f1": {
            "mean": mean_f1,
            "ci_95": ci_95_f1,
        },
        "statistical_test": {
            "null_hypothesis": "True Recall < 0.85",
            "p_value": p_value_recall,
            "is_significant_05": p_value_recall < 0.05,
            "is_significant_01": p_value_recall < 0.01,
        },
    }
    
    os.makedirs("./reports", exist_ok=True)
    with open("./reports/bootstrap_statistical_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved bootstrap results to ./reports/bootstrap_statistical_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
