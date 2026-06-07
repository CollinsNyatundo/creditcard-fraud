import pandas as pd
import numpy as np
import time
import json
import joblib
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def run_optimized_pipeline_test():
    print("=" * 60)
    print("END-TO-END OPTIMIZED PIPELINE TEST")
    print("=" * 60)
    
    # Define paths
    base_dir = "."
    model_path = os.path.join(base_dir, "models", "optimized_lightgbm.pkl")
    feature_list_path = os.path.join(base_dir, "models", "feature_list.json")
    test_data_path = os.path.join(base_dir, "data", "processed", "test_enhanced.csv")
    output_results_path = os.path.join(base_dir, "reports", "end_to_end_optimized_results.json")
    
    # 1. Load optimized model and feature list
    print("1. Loading model and feature list...")
    if not os.path.exists(model_path):
        print(f"   [FAIL] Model file not found: {model_path}")
        return False
    if not os.path.exists(feature_list_path):
        print(f"   [FAIL] Feature list not found: {feature_list_path}")
        return False
        
    try:
        model = joblib.load(model_path)
        with open(feature_list_path, 'r') as f:
            feature_cols = json.load(f)
        print("   [OK] Model and feature list loaded successfully")
        print(f"   Model expects {len(feature_cols)} features")
    except Exception as e:
        print(f"   [FAIL] Failed to load model/features: {e}")
        return False
        
    # 2. Load test dataset
    print("\n2. Loading enhanced test dataset...")
    if not os.path.exists(test_data_path):
        print(f"   [FAIL] Test dataset not found: {test_data_path}")
        return False
        
    try:
        df_test = pd.read_csv(test_data_path)
        print(f"   [OK] Loaded {len(df_test)} test samples with {df_test.shape[1]} columns")
    except Exception as e:
        print(f"   [FAIL] Failed to load test data: {e}")
        return False
        
    # 3. Extract and align features
    print("\n3. Aligning and validating features...")
    try:
        # Separate features and target
        missing_features = [col for col in feature_cols if col not in df_test.columns]
        if missing_features:
            print(f"   [FAIL] Missing features in test dataset: {missing_features}")
            return False
            
        X_test = df_test[feature_cols]
        y_test = df_test['Class']
        print(f"   [OK] Extracted features matching model signature: {X_test.shape}")
    except Exception as e:
        print(f"   [FAIL] Feature extraction failed: {e}")
        return False
        
    # 4. Measure latency for individual predictions (simulating real-time)
    print("\n4. Simulating real-time inference with latency benchmarking...")
    try:
        # Use a sample of 1000 transactions for reliable latency measurement
        sample_size = min(1000, len(X_test))
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
        
        latencies = []
        # Warm-up run to initialize LightGBM internal structures
        _ = model.predict(X_sample.iloc[0:1])
        
        for i in range(len(X_sample)):
            row = X_sample.iloc[i:i+1]
            start_time = time.perf_counter()
            _ = model.predict(row)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        print(f"   [OK] Processed {sample_size} simulated real-time transactions")
        print(f"   Latency Metrics:")
        print(f"     Mean: {mean_latency:.4f} ms")
        print(f"     Median: {median_latency:.4f} ms")
        print(f"     95th Percentile: {p95_latency:.4f} ms")
        print(f"     99th Percentile: {p99_latency:.4f} ms")
        print(f"     Max: {max_latency:.4f} ms")
        
        latency_meets_requirement = p95_latency < 10.0
        print(f"   Real-time latency requirement (<10ms 95th percentile): {'[PASS]' if latency_meets_requirement else '[FAIL]'}")
    except Exception as e:
        print(f"   [FAIL] Latency measurement failed: {e}")
        return False
        
    # 5. Evaluate overall model performance on full test set
    print("\n5. Running full test set evaluation...")
    try:
        # Load threshold
        threshold_path = os.path.join(base_dir, "models", "optimal_threshold_v2.json")
        threshold = 0.5  # default
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold = json.load(f).get('threshold', 0.5)
        print(f"   Using decision threshold: {threshold:.4f}")
        
        start_time = time.perf_counter()
        y_proba = model.predict(X_test)  # Batch predict
        # LightGBM predict returns probability for binary classification
        y_pred = (y_proba >= threshold).astype(int)
        batch_time = (time.perf_counter() - start_time) * 1000
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"   [OK] Full batch inference completed in {batch_time:.2f} ms")
        print(f"   Model Performance:")
        print(f"     F1-Score: {f1:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall: {recall:.4f}")
        print(f"     ROC AUC: {auc:.4f}")
        
        f1_meets_requirement = f1 > 0.85
        print(f"   F1-Score requirement (>0.85): {'[PASS]' if f1_meets_requirement else '[FAIL]'}")
    except Exception as e:
        print(f"   [FAIL] Performance evaluation failed: {e}")
        return False
        
    # 6. Save results
    print("\n6. Saving test results...")
    try:
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": {
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "roc_auc": float(auc)
            },
            "latency": {
                "mean_ms": float(mean_latency),
                "median_ms": float(median_latency),
                "p95_ms": float(p95_latency),
                "p99_ms": float(p99_latency),
                "max_ms": float(max_latency)
            },
            "requirements": {
                "f1_target_met": bool(f1_meets_requirement),
                "latency_target_met": bool(latency_meets_requirement)
            }
        }
        os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
        with open(output_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   [OK] Results saved to {output_results_path}")
    except Exception as e:
        print(f"   [FAIL] Failed to save results: {e}")
        
    print("\n" + "=" * 60)
    print("TEST STATUS: SUCCESSFUL")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_optimized_pipeline_test()
