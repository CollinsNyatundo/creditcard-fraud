import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
import lightgbm as lgb

def load_model_and_preprocessor():
    """Load the baseline LightGBM model and preprocessor"""
    print("Loading model and preprocessor...")
    model = joblib.load('./models/baseline_lightgbm.pkl')
    preprocessor = joblib.load('./models/preprocessor.pkl')
    # Load optimal threshold
    with open('./models/optimal_threshold.json', 'r') as f:
        optimal_threshold = json.load(f)['threshold']
    print(f"Model loaded successfully. Optimal threshold: {optimal_threshold}")
    return model, preprocessor, optimal_threshold

def load_test_data():
    """Load and prepare test data"""
    print("Loading test data...")
    test_df = pd.read_csv('./data/processed/test.csv')
    print(f"Test data loaded. Shape: {test_df.shape}")
    # Separate features and target
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    return X_test, y_test

def prepare_features(X_test):
    """Align test features with the model's signature using feature_names.json"""
    X_test = X_test.copy()

    # Load expected feature names
    with open('./models/feature_names.json', 'r') as f:
        expected_features = json.load(f)['feature_names']

    # One-hot encode the 'Amount_Bin' column
    if 'Amount_Bin' in X_test.columns:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        amount_bin_encoded = encoder.fit_transform(X_test[['Amount_Bin']])
        feature_names = encoder.get_feature_names_out(['Amount_Bin'])
        amount_bin_df = pd.DataFrame(amount_bin_encoded, columns=feature_names, index=X_test.index)

        # Drop original Amount_Bin and boolean columns if they exist
        columns_to_drop = ['Amount_Bin'] + [col for col in X_test.columns if col in ['Amount_Low', 'Amount_Medium', 'Amount_High']]
        X_test_processed = X_test.drop(columns=columns_to_drop)
        X_test_final = pd.concat([X_test_processed, amount_bin_df], axis=1)
    else:
        X_test_final = X_test.copy()

    # Fill in any missing expected features with 0.0
    for col in expected_features:
        if col not in X_test_final.columns:
            X_test_final[col] = 0.0

    # Select and order columns to match the model's signature
    X_test_final = X_test_final[expected_features]
    return X_test_final

def evaluate_model_performance(model, X_test_final, y_test, threshold):
    """Evaluate model performance on test set"""
    print("Evaluating model performance...")
    # Get predictions using Booster's predict (returns probabilities directly)
    y_pred_proba = model.predict(X_test_final, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Detailed classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),  # Convert to list for JSON serialization
        'classification_report': class_report
    }
    print(f"Model Performance Metrics:")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return metrics, y_pred_proba

def benchmark_inference_latency(model, X_test_final, sample_size=1000):
    """Benchmark inference latency"""
    print("Benchmarking inference latency...")
    # Take a sample of transactions for latency testing
    sample_indices = np.random.choice(len(X_test_final), min(sample_size, len(X_test_final)), replace=False)
    X_sample = X_test_final.iloc[sample_indices]

    latencies = []
    # Warm-up run to initialize LightGBM internal structures
    _ = model.predict(X_sample.iloc[0:1], num_iteration=model.best_iteration)

    # Measure latency for each prediction
    for i in range(len(X_sample)):
        start_time = time.perf_counter()
        _ = model.predict(X_sample.iloc[i:i + 1], num_iteration=model.best_iteration)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency_ms)

    # Calculate statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    max_latency = np.max(latencies)

    latency_metrics = {
        'mean_latency_ms': mean_latency,
        'median_latency_ms': median_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'max_latency_ms': max_latency
    }
    print(f"Latency Metrics (ms):")
    print(f"  Mean: {mean_latency:.4f}")
    print(f"  Median: {median_latency:.4f}")
    print(f"  95th Percentile: {p95_latency:.4f}")
    print(f"  99th Percentile: {p99_latency:.4f}")
    print(f"  Max: {max_latency:.4f}")
    return latency_metrics

def deployment_readiness_assessment(metrics, latency_metrics):
    """Assess deployment readiness based on requirements"""
    print("\nDeployment Readiness Assessment:")
    f1_score = metrics['f1_score']
    p95_latency = latency_metrics['p95_latency_ms']

    # Check F1-score requirement (>0.85)
    f1_requirement_met = f1_score > 0.85
    print(f"F1-Score Requirement (>0.85): {'PASS' if f1_requirement_met else 'FAIL'}")
    print(f"  Current F1-Score: {f1_score:.4f}")
    print(f"  Gap to target: {0.85 - f1_score:.4f}")

    # Check latency requirement (<10ms for 95th percentile)
    latency_requirement_met = p95_latency < 10.0
    print(f"Latency Requirement (<10ms 95th percentile): {'PASS' if latency_requirement_met else 'FAIL'}")
    print(f"  Current 95th Percentile: {p95_latency:.4f}ms")
    print(f"  Gap to target: {p95_latency - 10.0:.4f}ms")

    # Overall assessment
    overall_ready = f1_requirement_met and latency_requirement_met
    print(f"\nOverall Deployment Ready: {'YES' if overall_ready else 'NO'}")
    assessment = {
        'f1_requirement_met': f1_requirement_met,
        'latency_requirement_met': latency_requirement_met,
        'overall_ready': overall_ready,
        'f1_gap': 0.85 - f1_score,
        'latency_gap': p95_latency - 10.0
    }
    return assessment

def save_evaluation_results(metrics, latency_metrics, assessment):
    """Save evaluation results to JSON file"""
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    results = {
        'model_performance': convert_types(metrics),
        'latency_benchmark': convert_types(latency_metrics),
        'deployment_assessment': convert_types(assessment)
    }
    with open('./reports/final_performance_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to ./reports/final_performance_evaluation.json")

def main():
    """Main evaluation function"""
    print("=== Final Model Evaluation and Deployment Readiness Assessment ===\n")
    # Load model and data
    model, preprocessor, threshold = load_model_and_preprocessor()
    X_test, y_test = load_test_data()

    # Align and prepare test features
    X_test_final = prepare_features(X_test)

    # Run post-training feature explanations coverage check
    from model.src.feature_coverage_check import check_feature_explanations_coverage
    print("Running post-training feature explanations coverage check...")
    check_feature_explanations_coverage(list(X_test_final.columns))

    # Evaluate model performance
    metrics, y_pred_proba = evaluate_model_performance(model, X_test_final, y_test, threshold)

    # Benchmark inference latency
    latency_metrics = benchmark_inference_latency(model, X_test_final)

    # Assess deployment readiness
    assessment = deployment_readiness_assessment(metrics, latency_metrics)

    # Save results
    save_evaluation_results(metrics, latency_metrics, assessment)
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
