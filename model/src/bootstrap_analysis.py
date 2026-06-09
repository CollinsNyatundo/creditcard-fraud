import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.special import expit

def main():
    print("=" * 60)
    print("BOOTSTRAP STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Load model and configurations
    model_path = "./models/optimized_lightgbm.pkl"
    feature_list_path = "./models/feature_list.json"
    threshold_path = "./models/optimal_threshold_v2.json"
    
    if not os.path.exists(model_path) or not os.path.exists(feature_list_path) or not os.path.exists(threshold_path):
        print("[FAIL] Optimized model or configs not found. Run training first.")
        return
        
    model = joblib.load(model_path)
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
    
    # Predict probabilities on full test set
    raw_preds = model.predict(X_test)
    if is_focal_loss:
        y_proba = expit(raw_preds + init_score)
    else:
        y_proba = raw_preds
        
    y_pred = (y_proba >= threshold).astype(int)
    
    # Original metrics
    orig_f1 = f1_score(y_test, y_pred)
    orig_precision = precision_score(y_test, y_pred)
    orig_recall = recall_score(y_test, y_pred)
    orig_roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Original F1: {orig_f1:.4f}")
    print(f"Original Precision: {orig_precision:.4f}")
    print(f"Original Recall: {orig_recall:.4f}")
    print(f"Original ROC AUC: {orig_roc_auc:.4f}")
    
    # Run Bootstrap
    print("Running bootstrap resampling (B = 10,000)...")
    np.random.seed(42)
    B = 10000
    bootstrap_f1_scores = []
    
    n_samples = len(y_test)
    
    for i in range(B):
        # Sample indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_b = y_test[indices]
        y_proba_b = y_proba[indices]
        y_pred_b = (y_proba_b >= threshold).astype(int)
        
        f1_b = f1_score(y_true_b, y_pred_b)
        bootstrap_f1_scores.append(f1_b)
        
    bootstrap_f1_scores = np.array(bootstrap_f1_scores)
    mean_f1 = float(np.mean(bootstrap_f1_scores))
    median_f1 = float(np.median(bootstrap_f1_scores))
    ci_95 = [float(np.percentile(bootstrap_f1_scores, 2.5)), float(np.percentile(bootstrap_f1_scores, 97.5))]
    
    # P-value for null hypothesis F1 < 0.85
    p_value = float(np.mean(bootstrap_f1_scores < 0.85))
    
    print(f"Bootstrap Mean F1: {mean_f1:.4f}")
    print(f"Bootstrap Median F1: {median_f1:.4f}")
    print(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"p-value (F1 < 0.85): {p_value:.4f}")
    
    results = {
        "original_f1": orig_f1,
        "original_precision": orig_precision,
        "original_recall": orig_recall,
        "original_roc_auc": orig_roc_auc,
        "bootstrap_f1": {
            "mean": mean_f1,
            "median": median_f1,
            "ci_95": ci_95
        },
        "statistical_test": {
            "null_hypothesis": "True F1-score < 0.85",
            "p_value": p_value,
            "is_significant_05": p_value < 0.05,
            "is_significant_01": p_value < 0.01
        }
    }
    
    os.makedirs("./reports", exist_ok=True)
    with open("./reports/bootstrap_statistical_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved bootstrap results to ./reports/bootstrap_statistical_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
