import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import OneHotEncoder
import time
import joblib
import json
import os
from datetime import datetime
# Create directories if they don't exist
os.makedirs('./models', exist_ok=True)
os.makedirs('./reports', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
def load_data():
    """Load the processed datasets."""
    print("Loading datasets...")
    # Load balanced training set
    train_df = pd.read_csv('./data/processed/train_balanced.csv')
    # Load validation set
    val_df = pd.read_csv('./data/processed/val.csv')
    # Load test set
    test_df = pd.read_csv('./data/processed/test.csv')
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    return train_df, val_df, test_df
def prepare_features(train_df, val_df, test_df):
    """Prepare features for training with proper encoding of categorical variables."""
    print("\nPreparing features...")
    # For the training set, it's already balanced and processed
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    # For validation and test sets, we need to handle the 'Amount_Bin' column
    X_val = val_df.drop('Class', axis=1)
    y_val = val_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    # Handle categorical variables in validation and test sets
    # One-hot encode the 'Amount_Bin' column
    if 'Amount_Bin' in X_val.columns:
        # Create one-hot encoder
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Fit encoder on validation set (since it represents the original distribution)
        amount_bin_encoded = encoder.fit_transform(X_val[['Amount_Bin']])
        # Get feature names
        feature_names = encoder.get_feature_names_out(['Amount_Bin'])
        # Create DataFrame with encoded features
        amount_bin_df = pd.DataFrame(amount_bin_encoded, columns=feature_names, index=X_val.index)
        # Drop original 'Amount_Bin' and boolean columns, add encoded ones
        columns_to_drop = ['Amount_Bin'] + [col for col in X_val.columns if col in ['Amount_Low', 'Amount_Medium', 'Amount_High']]
        X_val_processed = X_val.drop(columns=columns_to_drop)
        X_val_final = pd.concat([X_val_processed, amount_bin_df], axis=1)
        # Do the same for test set
        amount_bin_encoded_test = encoder.transform(X_test[['Amount_Bin']])
        amount_bin_df_test = pd.DataFrame(amount_bin_encoded_test, columns=feature_names, index=X_test.index)
        X_test_processed = X_test.drop(columns=columns_to_drop)
        X_test_final = pd.concat([X_test_processed, amount_bin_df_test], axis=1)
        # For training set, we need to align the columns
        # Drop any boolean columns if they exist in training set
        columns_to_drop_train = [col for col in X_train.columns if col in ['Amount_Low', 'Amount_Medium', 'Amount_High']]
        if columns_to_drop_train:
            X_train_final = X_train.drop(columns=columns_to_drop_train)
        else:
            X_train_final = X_train
        # Ensure all datasets have the same columns
        # Get common columns
        common_columns = set(X_train_final.columns) & set(X_val_final.columns) & set(X_test_final.columns)
        # Remove 'Amount_Bin' from common columns if it exists
        common_columns = [col for col in common_columns if col != 'Amount_Bin']
        print(f"Common columns: {len(common_columns)}")
        # Select only common columns
        X_train_final = X_train_final[common_columns]
        X_val_final = X_val_final[common_columns]
        X_test_final = X_test_final[common_columns]
    else:
        # If 'Amount_Bin' is not present, just ensure all datasets have the same columns
        common_columns = set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
        common_columns = [col for col in common_columns if col != 'Amount_Bin']
        X_train_final = X_train[common_columns]
        X_val_final = X_val[common_columns]
        X_test_final = X_test[common_columns]
    print(f"Training features shape: {X_train_final.shape}")
    print(f"Validation features shape: {X_val_final.shape}")
    print(f"Test features shape: {X_test_final.shape}")
    # Check for any non-numeric columns
    non_numeric_cols = X_train_final.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found: {non_numeric_cols}")
        # Try to convert them to numeric
        for col in non_numeric_cols:
            X_train_final[col] = pd.to_numeric(X_train_final[col], errors='coerce')
            X_val_final[col] = pd.to_numeric(X_val_final[col], errors='coerce')
            X_test_final[col] = pd.to_numeric(X_test_final[col], errors='coerce')
    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test
def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with CPU optimization."""
    print("\n--- TRAINING LIGHTGBM MODEL ---")
    # Check for any non-numeric columns
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Error: Non-numeric columns found in training data: {non_numeric_cols}")
        return None
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    # Define parameters for CPU training
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'device_type': 'cpu',  # Force CPU usage
        'random_state': 42
    }
    # Train the model
    print("Training LightGBM model...")
    start_time = time.time()
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best iteration: {model.best_iteration}")
    return model
def optimize_threshold(model, X_val, y_val):
    """Optimize decision threshold for F1-score."""
    print("\n--- OPTIMIZING DECISION THRESHOLD ---")
    # Get predicted probabilities
    y_proba = model.predict(X_val, num_iteration=model.best_iteration)
    # Calculate metrics for different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    optimal_precision = precision_scores[optimal_idx]
    optimal_recall = recall_scores[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Optimal F1-score: {optimal_f1:.4f}")
    print(f"Precision at optimal threshold: {optimal_precision:.4f}")
    print(f"Recall at optimal threshold: {optimal_recall:.4f}")
    # Also calculate PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"PR-AUC: {pr_auc:.4f}")
    return optimal_threshold, {
        'threshold': float(optimal_threshold),
        'f1_score': float(optimal_f1),
        'precision': float(optimal_precision),
        'recall': float(optimal_recall),
        'pr_auc': float(pr_auc)
    }
def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate model on test set."""
    print("\n--- EVALUATING MODEL ON TEST SET ---")
    # Get predicted probabilities
    y_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_proba >= threshold).astype(int)
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    # Calculate PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC: {pr_auc:.4f}")
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
    print(f"Classification Report:\n{report}")
    return {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
def benchmark_latency(model, X_test, threshold, num_samples=1000):
    """Benchmark inference latency."""
    print("\n--- BENCHMARKING INFERENCE LATENCY ---")
    # Sample random transactions for benchmarking
    np.random.seed(42)
    sample_indices = np.random.choice(X_test.shape[0], min(num_samples, X_test.shape[0]), replace=False)
    sample_data = X_test.iloc[sample_indices]
    # Measure latency for individual predictions
    latencies = []
    for i in range(len(sample_data)):
        sample = sample_data.iloc[i:i + 1]  # Single sample
        start_time = time.perf_counter()
        proba = model.predict(sample, num_iteration=model.best_iteration)
        pred = (proba >= threshold).astype(int)
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)
    # Calculate statistics
    mean_latency = np.mean(latencies) * 1000  # Convert to milliseconds
    median_latency = np.median(latencies) * 1000
    p95_latency = np.percentile(latencies, 95) * 1000
    p99_latency = np.percentile(latencies, 99) * 1000
    max_latency = np.max(latencies) * 1000
    print(f"Mean latency: {mean_latency:.4f} ms")
    print(f"Median latency: {median_latency:.4f} ms")
    print(f"95th percentile latency: {p95_latency:.4f} ms")
    print(f"99th percentile latency: {p99_latency:.4f} ms")
    print(f"Max latency: {max_latency:.4f} ms")
    # Check if latency requirement is met
    if mean_latency < 10:
        print("[OK] Latency requirement (<10ms) met")
    else:
        print("[FAIL] Latency requirement (<10ms) not met")
    return {
        'mean_latency_ms': float(mean_latency),
        'median_latency_ms': float(median_latency),
        'p95_latency_ms': float(p95_latency),
        'p99_latency_ms': float(p99_latency),
        'max_latency_ms': float(max_latency),
        'latency_requirement_met': bool(mean_latency < 10)  # Convert to Python bool for JSON serialization
    }
def save_model_artifacts(model, threshold, feature_names):
    """Save model and related artifacts."""
    print("\n--- SAVING MODEL ARTIFACTS ---")
    # Save LightGBM model
    model_path = './models/baseline_lightgbm.txt'
    model.save_model(model_path)
    print(f"LightGBM model saved to: {model_path}")
    # Save model with joblib (alternative format)
    model_joblib_path = './models/baseline_lightgbm.pkl'
    joblib.dump(model, model_joblib_path)
    print(f"LightGBM model (joblib) saved to: {model_joblib_path}")
    # Save threshold
    threshold_path = './models/optimal_threshold.json'
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': float(threshold)}, f)  # Convert to Python float for JSON serialization
    print(f"Optimal threshold saved to: {threshold_path}")
    # Save feature names
    features_path = './models/feature_names.json'
    with open(features_path, 'w') as f:
        json.dump({'feature_names': feature_names.tolist()}, f)
    print(f"Feature names saved to: {features_path}")
    return model_path, model_joblib_path, threshold_path, features_path
def save_evaluation_results(evaluation_metrics, threshold_metrics, latency_metrics):
    """Save evaluation results to files."""
    print("\n--- SAVING EVALUATION RESULTS ---")
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):  # Handle boolean values
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    # Combine all metrics
    all_metrics = {
        'evaluation': convert_types(evaluation_metrics),
        'threshold_optimization': convert_types(threshold_metrics),
        'latency': convert_types(latency_metrics),
        'timestamp': datetime.now().isoformat()
    }
    # Save metrics
    metrics_path = './reports/model_evaluation.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")
    return metrics_path
def main():
    """Main function to train, tune, and evaluate the baseline model."""
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION - BASELINE MODEL TRAINING")
    print("=" * 70)
    # Load data
    train_df, val_df, test_df = load_data()
    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(train_df, val_df, test_df)
    # Check data types before training
    print("\n--- DATA TYPE CHECK ---")
    non_numeric_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_val = X_val.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_test = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_train or non_numeric_val or non_numeric_test:
        print(f"Non-numeric columns in training data: {non_numeric_train}")
        print(f"Non-numeric columns in validation data: {non_numeric_val}")
        print(f"Non-numeric columns in test data: {non_numeric_test}")
        print("Attempting to convert to numeric...")
        # Convert to numeric
        for col in non_numeric_train:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        for col in non_numeric_val:
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
        for col in non_numeric_test:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    else:
        print("All columns are numeric. Ready for training.")
    # Train model
    model = train_lightgbm_model(X_train, y_train, X_val, y_val)
    if model is None:
        print("Model training failed due to data issues.")
        return None, None
    # Optimize threshold
    optimal_threshold, threshold_metrics = optimize_threshold(model, X_val, y_val)
    # Evaluate on test set
    evaluation_metrics = evaluate_model(model, X_test, y_test, optimal_threshold)
    # Benchmark latency
    latency_metrics = benchmark_latency(model, X_test, optimal_threshold)
    # Save model artifacts
    model_paths = save_model_artifacts(model, optimal_threshold, X_train.columns)
    # Save evaluation results
    metrics_path = save_evaluation_results(evaluation_metrics, threshold_metrics, latency_metrics)
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("=" * 70)
    # Print summary
    print("\n--- SUMMARY ---")
    print(f"Final F1-score (test): {evaluation_metrics['f1_score']:.4f}")
    print(f"Final Precision (test): {evaluation_metrics['precision']:.4f}")
    print(f"Final Recall (test): {evaluation_metrics['recall']:.4f}")
    print(f"Mean latency: {latency_metrics['mean_latency_ms']:.4f} ms")
    print(f"Latency requirement met: {'Yes' if latency_metrics['latency_requirement_met'] else 'No'}")
    return model, optimal_threshold
if __name__ == "__main__":
    model, threshold = main()
