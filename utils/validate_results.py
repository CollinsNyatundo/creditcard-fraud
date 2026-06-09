import pandas as pd
import json
import os
def validate_artifacts():
    """Validate that all required artifacts were created"""
    print("Validating feature engineering and hyperparameter tuning results...")
    print("=" * 60)
    # Check enhanced datasets
    print("1. Checking enhanced datasets:")
    datasets = {
        "train_enhanced.csv (Advanced)": "./data/processed/train_enhanced.csv",
        "val_enhanced.csv (Advanced)": "./data/processed/val_enhanced.csv",
        "test_enhanced.csv (Advanced)": "./data/processed/test_enhanced.csv",
        "train_enhanced_lite.csv (Lightweight)": "./data/processed/train_enhanced_lite.csv",
        "val_enhanced_lite.csv (Lightweight)": "./data/processed/val_enhanced_lite.csv",
        "test_enhanced_lite.csv (Lightweight)": "./data/processed/test_enhanced_lite.csv",
        "train_enhanced_minimal.csv (Minimal)": "./data/processed/train_enhanced_minimal.csv",
        "val_enhanced_minimal.csv (Minimal)": "./data/processed/val_enhanced_minimal.csv",
        "test_enhanced_minimal.csv (Minimal)": "./data/processed/test_enhanced_minimal.csv"
    }
    for name, path in datasets.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"   [OK] {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"   [FAIL] {name}: NOT FOUND")
    # Check model artifacts
    print("\n2. Checking model artifacts:")
    artifacts = {
        "optimized_lightgbm.pkl": "./models/optimized_lightgbm.pkl",
        "feature_list.json": "./models/feature_list.json",
        "optimal_threshold_v2.json": "./models/optimal_threshold_v2.json"
    }
    for name, path in artifacts.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   [OK] {name}: {size} bytes")
        else:
            print(f"   [FAIL] {name}: NOT FOUND")
    # Check reports
    print("\n3. Checking reports:")
    reports = {
        "hyperparameter_optimization.json": "./reports/hyperparameter_optimization.json",
        "end_to_end_optimized_results.json": "./reports/end_to_end_optimized_results.json"
    }
    for name, path in reports.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   [OK] {name}: {size} bytes")
            # Validate versioning metadata tags
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                schema_version = data.get('schema_version')
                source_script = data.get('source_script')
                generated_at = data.get('generated_at') or data.get('timestamp')
                if schema_version and source_script and generated_at:
                    print(f"        [OK] Metadata tags validated (schema_v={schema_version}, script={source_script}, time={generated_at})")
                else:
                    print(f"        [WARNING] Missing metadata tags in {name} (schema_v={schema_version}, script={source_script}, time={generated_at})")
            except Exception as e:
                print(f"        [FAIL] Error validating metadata: {e}")
        else:
            print(f"   [FAIL] {name}: NOT FOUND")
    # Load and display feature list
    print("\n4. Feature engineering summary:")
    try:
        with open("./models/feature_list.json", "r") as f:
            features = json.load(f)
        print(f"   Total engineered features: {len(features)}")
        print(f"   First 10 features: {features[:10]}")
        print(f"   Last 10 features: {features[-10:]}")
    except Exception as e:
        print(f"   Error loading feature list: {e}")
    # Load and display optimization results
    print("\n5. Hyperparameter optimization summary:")
    try:
        with open("./reports/hyperparameter_optimization.json", "r") as f:
            results = json.load(f)
        if "final_metrics" in results:
            metrics = results["final_metrics"]
            print("   Final Metrics:")
            print(f"     F1 Score: {metrics.get('f1_score', 'N/A')}")
            print(f"     Precision: {metrics.get('precision', 'N/A')}")
            print(f"     Recall: {metrics.get('recall', 'N/A')}")
            print(f"     ROC AUC: {metrics.get('roc_auc', 'N/A')}")
            print(f"     Optimal Threshold: {metrics.get('optimal_threshold', 'N/A')}")

            # Print latency if present in final_metrics
            avg_lat = metrics.get('avg_latency')
            p95_lat = metrics.get('latency_95_percentile')
            print("   Latency Benchmark:")
            if avg_lat is not None:
                print(f"     Mean Single Transaction: {avg_lat * 1000:.4f} ms")
            if p95_lat is not None:
                print(f"     95th Percentile: {p95_lat * 1000:.4f} ms")
        elif "validation_metrics" in results:
            metrics = results["validation_metrics"]
            print("   Validation Metrics:")
            print(f"     F1 Score: {metrics.get('f1_score', 'N/A')}")
            print(f"     Precision: {metrics.get('precision', 'N/A')}")
            print(f"     Recall: {metrics.get('recall', 'N/A')}")
            print(f"     ROC AUC: {metrics.get('roc_auc', 'N/A')}")
            print(f"     Optimal Threshold: {metrics.get('optimal_threshold', 'N/A')}")
        if "latency_benchmark" in results:
            latency = results["latency_benchmark"]
            print("   Latency Benchmark:")
            print(f"     Mean Single Transaction: {latency.get('mean_single_latency_ms', 'N/A')} ms")
            print(f"     95th Percentile: {latency.get('percentile_95_single_latency_ms', 'N/A')} ms")
        if "optuna_study" in results:
            study = results["optuna_study"]
            print(f"   Best Optimization Value: {study.get('best_value', 'N/A')}")
    except Exception as e:
        print(f"   Error loading optimization results: {e}")
    print("\n" + "=" * 60)
    print("Validation complete!")
if __name__ == "__main__":
    validate_artifacts()
