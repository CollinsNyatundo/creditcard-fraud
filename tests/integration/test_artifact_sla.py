import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import pytest

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_model_artifact_files_exist():
    """Verify that all optimized model artifacts exist and are non-empty."""
    required_files = [
        './models/optimized_lightgbm.pkl',
        './models/feature_list.json',
        './models/optimal_threshold_v2.json'
    ]
    for file_path in required_files:
        assert os.path.exists(file_path), f"Artifact file {file_path} does not exist."
        assert os.path.getsize(file_path) > 0, f"Artifact file {file_path} is empty."

def test_feature_column_alignment_and_types():
    """Verify feature list aligns with test dataset and has correct types."""
    # Load feature list
    with open('./models/feature_list.json', 'r') as f:
        feature_list = json.load(f)
    
    assert isinstance(feature_list, list), "feature_list.json should contain a list of feature names"
    assert len(feature_list) > 0, "feature_list should not be empty"

    # Load test dataset
    test_path = './data/processed/test_enhanced.csv'
    assert os.path.exists(test_path), f"Processed test dataset {test_path} is missing."
    test_df = pd.read_csv(test_path)

    # Check that all features in feature_list exist in the test dataset
    for col in feature_list:
        assert col in test_df.columns, f"Feature '{col}' in feature list is missing from processed test data."
        # Assert columns are numeric
        assert pd.api.types.is_numeric_dtype(test_df[col]), f"Feature '{col}' is not numeric."

def test_inference_latency_sla():
    """Assert that the 95th percentile latency of single-row inference is less than 10ms."""
    # Load model
    model = joblib.load('./models/optimized_lightgbm.pkl')
    
    # Load feature list
    with open('./models/feature_list.json', 'r') as f:
        feature_list = json.load(f)

    # Load test dataset
    test_df = pd.read_csv('./data/processed/test_enhanced.csv')
    X_test = test_df[feature_list]

    # Warm-up run to initialize LightGBM internal structures
    warmup_row = X_test.iloc[0:1]
    _ = model.predict(warmup_row)

    # Performance latency benchmark over 1000 single predictions
    sample_size = min(1000, len(X_test))
    latencies = []
    
    # Measure latency for single rows
    for i in range(sample_size):
        row = X_test.iloc[i:i+1]
        start_time = time.perf_counter()
        _ = model.predict(row)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # milliseconds

    latencies = np.array(latencies)
    p95_latency = np.percentile(latencies, 95)
    mean_latency = np.mean(latencies)

    print(f"\nInference Latency Benchmark (n={sample_size}):")
    print(f"  Mean Latency: {mean_latency:.4f} ms")
    print(f"  p95 Latency: {p95_latency:.4f} ms")

    # Assert p95 latency is < 10ms SLA
    assert p95_latency < 10.0, f"p95 latency of {p95_latency:.2f}ms exceeds the 10ms SLA target."
