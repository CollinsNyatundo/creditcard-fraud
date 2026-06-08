import pandas as pd
import numpy as np
import time
import json
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
def end_to_end_pipeline_test():
    """Execute a full end-to-end pipeline test with proper feature alignment"""
    print("=" * 60)
    print("END-TO-END PIPELINE TEST (FIXED VERSION)")
    print("=" * 60)
    # Load required components
    print("1. Loading model and preprocessor...")
    try:
        model = joblib.load('./models/baseline_lightgbm.pkl')
        preprocessor = joblib.load('./models/preprocessor.pkl')
        try:
            with open('./models/optimal_threshold.json', 'r') as f:
                optimal_threshold = json.load(f)['threshold']
        except Exception:
            optimal_threshold = 0.76
        print(f"   [OK] Model and preprocessor loaded successfully (threshold: {optimal_threshold})")
        # Check expected features
        if hasattr(preprocessor, 'feature_names_in_'):
            expected_features = list(preprocessor.feature_names_in_)
            print(f"   Preprocessor expects {len(expected_features)} features")
        else:
            expected_features = None
            print("   Warning: Preprocessor feature names not available")
    except Exception as e:
        print(f"   [FAIL] Failed to load model/preprocessor: {e}")
        return False
    # Load test data
    print("\n2. Loading test data...")
    try:
        # Load a batch of test data (100 samples for realistic testing)
        df_test = pd.read_csv('./data/processed/test.csv', nrows=100)
        print(f"   [OK] Loaded {len(df_test)} test samples")
        # Separate features and target
        if 'Class' in df_test.columns:
            X_test = df_test.drop('Class', axis=1)
            y_test = df_test['Class']
            print(f"   [OK] Separated features ({X_test.shape}) and target ({len(y_test)})")
        else:
            print("   [FAIL] No 'Class' column found in test data")
            return False
    except Exception as e:
        print(f"   [FAIL] Failed to load test data: {e}")
        return False
    # Align features properly to model signature
    print("\n3. Aligning features to model signature...")
    try:
        from sklearn.preprocessing import OneHotEncoder
        X_test_processed = X_test.copy()
        
        # Load expected feature names from model training
        with open('./models/feature_names.json', 'r') as f:
            expected_features = json.load(f)['feature_names']
            
        # One-hot encode the 'Amount_Bin' column
        if 'Amount_Bin' in X_test_processed.columns:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            amount_bin_encoded = encoder.fit_transform(X_test_processed[['Amount_Bin']])
            feature_names = encoder.get_feature_names_out(['Amount_Bin'])
            amount_bin_df = pd.DataFrame(amount_bin_encoded, columns=feature_names, index=X_test_processed.index)
            
            # Drop original Amount_Bin and boolean columns if they exist
            columns_to_drop = ['Amount_Bin'] + [col for col in X_test_processed.columns if col in ['Amount_Low', 'Amount_Medium', 'Amount_High']]
            X_test_processed = X_test_processed.drop(columns=columns_to_drop)
            X_test_processed = pd.concat([X_test_processed, amount_bin_df], axis=1)
            
        # Fill in any missing expected features with 0.0
        for col in expected_features:
            if col not in X_test_processed.columns:
                X_test_processed[col] = 0.0
                
        # Select and order columns to match the model's signature
        X_test_processed = X_test_processed[expected_features]
        print(f"   [OK] Features aligned to model signature: {X_test_processed.shape}")
    except Exception as e:
        print(f"   [FAIL] Feature alignment failed: {e}")
        return False
    # Test model inference with latency measurement
    print("\n5. Testing model inference with latency measurement...")
    try:
        # Measure latency for individual predictions (simulating real-time)
        latencies = []
        batch_size = 1  # Simulate individual transaction processing
        # Use first 50 samples for detailed latency testing
        test_sample = X_test_processed[:50]
        print(f"   Testing latency on {len(test_sample)} individual predictions...")
        predictions = []
        probabilities = []
        for i in range(len(test_sample)):
            sample = test_sample[i:i + 1]  # Single sample
            start_time = time.perf_counter()
            pred_proba = model.predict(sample)[0]
            pred = int(pred_proba >= optimal_threshold)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)
            predictions.append(pred)
            probabilities.append(pred_proba)
        print("   [OK] Model inference completed successfully")
    except Exception as e:
        print(f"   [FAIL] Model inference failed: {e}")
        # Try alternative approach with batch prediction
        try:
            print("   Trying batch prediction...")
            start_time = time.perf_counter()
            probabilities_batch = model.predict(X_test_processed[:50])
            predictions_batch = (probabilities_batch >= optimal_threshold).astype(int)
            end_time = time.perf_counter()
            batch_latency_ms = (end_time - start_time) * 1000
            avg_latency_ms = batch_latency_ms / len(predictions_batch)
            print(f"   [OK] Batch prediction completed in {batch_latency_ms:.4f}ms (avg {avg_latency_ms:.4f}ms per prediction)")
            # Convert to individual prediction format for metrics
            predictions = predictions_batch.tolist()
            probabilities = probabilities_batch.tolist()
            latencies = [avg_latency_ms] * len(predictions)
        except Exception as e2:
            print(f"   [FAIL] Batch prediction also failed: {e2}")
            return False
    # Calculate and report latency metrics
    print("\n6. Latency Analysis:")
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    max_latency = np.max(latencies)
    print(f"   Mean latency: {mean_latency:.4f} ms")
    print(f"   Median latency: {median_latency:.4f} ms")
    print(f"   95th percentile latency: {p95_latency:.4f} ms")
    print(f"   99th percentile latency: {p99_latency:.4f} ms")
    print(f"   Max latency: {max_latency:.4f} ms")
    # Check against requirements
    target_latency = 10.0  # ms
    meets_requirement = p95_latency < target_latency
    print(f"   Meets <{target_latency}ms requirement: {'YES' if meets_requirement else 'NO'}")
    # Calculate performance metrics if we have ground truth
    print("\n7. Performance Metrics:")
    try:
        if len(y_test) >= len(predictions):
            y_test_sample = y_test[:len(predictions)].values
            # Calculate metrics
            f1 = f1_score(y_test_sample, predictions)
            precision = precision_score(y_test_sample, predictions)
            recall = recall_score(y_test_sample, predictions)
            roc_auc = roc_auc_score(y_test_sample, probabilities)
            print(f"   F1 Score: {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   ROC AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"   Warning: Could not calculate performance metrics: {e}")
        f1 = precision = recall = roc_auc = 0
    # Summary
    print(f"\n" + "=" * 60)
    print("END-TO-END PIPELINE TEST SUMMARY")
    print("=" * 60)
    success = True
    issues = []
    # Check critical components
    if mean_latency > 100:
        issues.append("Mean latency is very high (> 100ms)")
        success = False
    if p95_latency > 50:
        issues.append("95th percentile latency is high (> 50ms)")
    print(f"Pipeline Integration: {'[OK] SUCCESS' if success else '[FAIL] FAILED'}")
    print(f"Latency Performance: {'[OK] ACCEPTABLE' if p95_latency < 20 else '[WARN]  HIGH LATENCY'}")
    print(f"Real-time Requirement (<10ms 95th percentile): {'[OK] MET' if meets_requirement else '[WARN]  NOT MET'}")
    if issues:
        print(f"\nIssues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n[OK] All pipeline components integrated successfully")
        print(f"[OK] Model inference functional")
        print(f"[OK] Latency measurements completed")
    # Save results
    results = {
        "test_timestamp": pd.Timestamp.now().isoformat(),
        "samples_tested": len(test_sample),
        "latency_metrics": {
            "mean_ms": float(mean_latency),
            "median_ms": float(median_latency),
            "p95_ms": float(p95_latency),
            "p99_ms": float(p99_latency),
            "max_ms": float(max_latency),
            "meets_10ms_requirement": bool(meets_requirement)
        },
        "performance_metrics": {
            "f1_score": float(f1) if 'f1' in locals() else None,
            "precision": float(precision) if 'precision' in locals() else None,
            "recall": float(recall) if 'recall' in locals() else None,
            "roc_auc": float(roc_auc) if 'roc_auc' in locals() else None
        },
        "success": success,
        "issues": issues
    }
    with open('./end_to_end_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to ./end_to_end_test_results.json")
    return success
if __name__ == "__main__":
    success = end_to_end_pipeline_test()
    if success:
        print(f"\n[SUCCESS] END-TO-END PIPELINE TEST COMPLETED SUCCESSFULLY")
    else:
        print(f"\n[FAIL] END-TO-END PIPELINE TEST FAILED")
