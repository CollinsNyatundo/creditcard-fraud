#!/usr/bin/env python3
"""
Validation Script for Advanced Feature Engineering and Hyperparameter Tuning
This script validates that all components of the advanced feature engineering
and hyperparameter tuning pipeline have been successfully implemented.
"""
import os
import json
import pandas as pd
import pickle
import numpy as np
def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"[OK] {description} - EXISTS ({size} bytes)")
        return True
    else:
        print(f"[FAIL] {description} - MISSING")
        return False
def validate_enhanced_datasets():
    """Validate that enhanced datasets were created correctly"""
    print("Validating Enhanced Datasets...")
    print("-" * 40)
    datasets = [
        ("./data/processed/train_enhanced.csv", "Training Dataset"),
        ("./data/processed/val_enhanced.csv", "Validation Dataset"),
        ("./data/processed/test_enhanced.csv", "Test Dataset")
    ]
    all_valid = True
    for path, description in datasets:
        if check_file_exists(path, description):
            try:
                df = pd.read_csv(path)
                print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                # Check for Class column
                if 'Class' in df.columns:
                    fraud_count = df['Class'].sum()
                    total_count = len(df)
                    fraud_rate = fraud_count / total_count
                    print(f"  Fraud rate: {fraud_rate:.4f} ({fraud_count}/{total_count})")
                else:
                    print("  WARNING: No Class column found")
                    all_valid = False
            except Exception as e:
                print(f"  ERROR reading file: {e}")
                all_valid = False
        else:
            all_valid = False
        print()
    return all_valid
def validate_model_artifacts():
    """Validate that model artifacts were created correctly"""
    print("Validating Model Artifacts...")
    print("-" * 40)
    artifacts = [
        ("./models/optimized_lightgbm.pkl", "Optimized LightGBM Model"),
        ("./models/feature_list.json", "Feature List"),
        ("./models/optimal_threshold_v2.json", "Optimal Threshold")
    ]
    all_valid = True
    for path, description in artifacts:
        if check_file_exists(path, description):
            try:
                if path.endswith('.pkl'):
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"  Model type: {type(model)}")
                elif path.endswith('.json'):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"  Contains {len(data)} items")
                    elif isinstance(data, dict):
                        print(f"  Contains {len(data)} keys")
                    else:
                        print(f"  Data type: {type(data)}")
            except Exception as e:
                print(f"  ERROR reading file: {e}")
                all_valid = False
        else:
            all_valid = False
        print()
    return all_valid
def validate_reports():
    """Validate that reports were created correctly"""
    print("Validating Reports...")
    print("-" * 40)
    reports = [
        ("./reports/hyperparameter_optimization.json", "Hyperparameter Optimization Report")
    ]
    all_valid = True
    for path, description in reports:
        if check_file_exists(path, description):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"  Contains {len(data)} top-level keys")
                if 'validation_metrics' in data:
                    metrics = data['validation_metrics']
                    print(f"  F1 Score: {metrics.get('f1_score', 'N/A')}")
                    print(f"  Precision: {metrics.get('precision', 'N/A')}")
                    print(f"  Recall: {metrics.get('recall', 'N/A')}")
            except Exception as e:
                print(f"  ERROR reading file: {e}")
                all_valid = False
        else:
            all_valid = False
        print()
    return all_valid
def validate_feature_engineering():
    """Validate feature engineering results"""
    print("Validating Feature Engineering...")
    print("-" * 40)
    # Check feature list
    feature_path = "./models/feature_list.json"
    if check_file_exists(feature_path, "Feature List"):
        try:
            with open(feature_path, 'r') as f:
                features = json.load(f)
            print(f"  Total features: {len(features)}")
            # Check for expected feature types
            temporal_features = [f for f in features if 'hour' in f or 'day' in f or 'time' in f]
            statistical_features = [f for f in features if 'mean' in f or 'std' in f or 'ratio' in f]
            interaction_features = [f for f in features if '_x_' in f or '_sq' in f]
            anomaly_features = [f for f in features if 'cum' in f or 'zscore' in f or 'is_' in f]
            print(f"  Temporal features: {len(temporal_features)}")
            print(f"  Statistical features: {len(statistical_features)}")
            print(f"  Interaction features: {len(interaction_features)}")
            print(f"  Anomaly features: {len(anomaly_features)}")
            # Check for original V features
            v_features = [f for f in features if f.startswith('V')]
            print(f"  Original V-features: {len(v_features)}")
            if len(features) >= 40:
                print("  [OK] Feature count meets target (>=40)")
            else:
                print("  [FAIL] Feature count below target (<40)")
        except Exception as e:
            print(f"  ERROR reading feature list: {e}")
            return False
    else:
        return False
    print()
    return True
def main():
    """Main validation function"""
    print("Advanced Feature Engineering & Hyperparameter Tuning Validation")
    print("=" * 65)
    print()
    # Run all validations
    validations = [
        ("Enhanced Datasets", validate_enhanced_datasets),
        ("Model Artifacts", validate_model_artifacts),
        ("Reports", validate_reports),
        ("Feature Engineering", validate_feature_engineering)
    ]
    results = {}
    for name, validation_func in validations:
        print(f"[{name}]")
        results[name] = validation_func()
        print()
    # Summary
    print("VALIDATION SUMMARY")
    print("=" * 20)
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    print()
    if all_passed:
        print("[SUCCESS] ALL VALIDATIONS PASSED!")
        print("The advanced feature engineering and hyperparameter tuning")
        print("pipeline has been successfully implemented.")
    else:
        print("[WARN]  SOME VALIDATIONS FAILED!")
        print("Please check the output above for details on what needs to be fixed.")
    return all_passed
if __name__ == "__main__":
    main()
