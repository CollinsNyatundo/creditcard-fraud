import pandas as pd
import numpy as np
import time
import json
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
def end_to_end_pipeline_test():
    """Execute a full end-to-end pipeline test"""
    print("=" * 60)
    print("END-TO-END PIPELINE TEST")
    print("=" * 60)
    # Load required components
    print("1. Loading model and preprocessor...")
    try:
        model = joblib.load('./models/baseline_lightgbm.pkl')
        preprocessor = joblib.load('./models/preprocessor.pkl')
        print("   [OK] Model and preprocessor loaded successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to load model/preprocessor: {e}")
        return False
    # Load test data
    print("\n2. Loading test data...")
    try:
        # Load a batch of test data (500 samples for realistic testing)
        df_test = pd.read_csv('./data/processed/test.csv', nrows=500)
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
    # Test preprocessing compatibility
    print("\n3. Testing preprocessing compatibility...")
    try:
        # Get feature names expected by preprocessor
        if hasattr(preprocessor, 'feature_names_in_'):
            expected_features = list(preprocessor.feature_names_in_)
            print(f"   Preprocessor expects {len(expected_features)} features")
            # Check if we have all expected features
            available_features = list(X_test.columns)
            missing_features = set(expected_features) - set(available_features)
            extra_features = set(available_features) - set(expected_features)
            if missing_features:
                print(f"   Warning: Missing features: {list(missing_features)[:5]}...")
            if extra_features:
                print(f"   Info: Extra features: {list(extra_features)[:5]}...")
            # Align features if needed
            X_test_aligned = X_test.reindex(columns=expected_features, fill_value=0)
            print(f"   [OK] Features aligned to preprocessor requirements: {X_test_aligned.shape}")
        else:
            X_test_aligned = X_test
            print("   [OK] Using original features (no alignment needed)")
    except Exception as e:
        print(f"   [FAIL] Preprocessing compatibility test failed: {e}")
        return False
    # Test preprocessing
    print("\n4. Applying preprocessing...")
    try:
        start_time = time.time()
        X_test_processed = preprocessor.transform(X_test_aligned)
        preprocessing_time = time.time() - start_time
        print(f"   [OK] Preprocessing completed in {preprocessing_time:.4f}s")
        print(f"   Processed data shape: {X_test_processed.shape}")
    except Exception as e:
        print(f"   [FAIL] Preprocessing failed: {e}")
        return False
    # Test model inference with latency measurement
    print("\n5. Testing model inference with latency measurement...")
    try:
        # Measure latency for individual predictions (simulating real-time)
        latencies = []
        batch_size = 1  # Simulate individual transaction processing
        # Take a smaller sample for detailed latency testing
        test_sample = X_test_processed[:100]  # First 100 samples
        print(f"   Testing latency on {len(test_sample)} individual predictions...")
        predictions = []
        probabilities = []
        for i in range(len(test_sample)):
            sample = test_sample[i:i + 1]  # Single sample
            start_time = time.perf_counter()
            pred = model.predict(sample)
            pred_proba = model.predict_proba(sample)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)
            predictions.append(pred[0])
            probabilities.append(pred_proba[0][1])  # Probability of positive class
        print("   [OK] Model inference completed successfully")
    except Exception as e:
        print(f"   [FAIL] Model inference failed: {e}")
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
            # Compare with reported metrics
            print("\n8. Comparison with Reported Metrics:")
            with open('./reports/model_evaluation.json', 'r') as f:
                reported_metrics = json.load(f)
            reported_f1 = reported_metrics['evaluation']['f1_score']
            reported_latency = reported_metrics['latency']['p95_latency_ms']
            print(f"   Reported F1 Score: {reported_f1:.4f}")
            print(f"   Achieved F1 Score: {f1:.4f}")
            print(f"   F1 Score Difference: {abs(reported_f1 - f1):.4f}")
            print(f"   Reported 95th Percentile Latency: {reported_latency:.2f} ms")
            print(f"   Achieved 95th Percentile Latency: {p95_latency:.2f} ms")
            print(f"   Latency Difference: {abs(reported_latency - p95_latency):.2f} ms")
    except Exception as e:
        print(f"   Warning: Could not calculate performance metrics: {e}")
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
    print(f"Pipeline Components: {'[OK] SUCCESS' if success else '[FAIL] FAILED'}")
    print(f"Latency Performance: {'[OK] ACCEPTABLE' if p95_latency < 20 else '[WARN]  HIGH LATENCY'}")
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
            "meets_10ms_requirement": meets_requirement
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
        print(f"\n❌ END-TO-END PIPELINE TEST FAILED")
