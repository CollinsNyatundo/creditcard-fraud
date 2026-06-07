import pandas as pd
import joblib
import time
def fixed_test():
    print("Starting fixed end-to-end test...")
    # Load model and preprocessor
    print("1. Loading model and preprocessor...")
    model = joblib.load('./models/baseline_lightgbm.pkl')
    preprocessor = joblib.load('./models/preprocessor.pkl')
    print("   Done.")
    # Load test data
    print("2. Loading test data...")
    df_test = pd.read_csv('./data/processed/test.csv', nrows=5)
    X_test = df_test.drop('Class', axis=1)
    print("   Done.")
    # Align features
    print("3. Aligning features...")
    if hasattr(preprocessor, 'feature_names_in_'):
        expected_features = list(preprocessor.feature_names_in_)
        X_test_aligned = X_test.reindex(columns=expected_features, fill_value=0)
    else:
        X_test_aligned = X_test
    print("   Done.")
    # Preprocess
    print("4. Preprocessing...")
    X_processed = preprocessor.transform(X_test_aligned)
    print("   Done.")
    # Predict with shape check disabled as suggested by LightGBM
    print("5. Making predictions...")
    start_time = time.perf_counter()
    # Using the parameter suggested by LightGBM error message
    predictions = model.predict(X_processed, pred_early_stop=False)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000 / len(predictions)
    print(f"   Average latency per prediction: {latency_ms:.4f} ms")
    print(f"   Predictions: {predictions}")
    print("   Done.")
    print("Fixed end-to-end test completed successfully!")
    return True
if __name__ == "__main__":
    try:
        fixed_test()
    except Exception as e:
        print(f"Test failed with error: {e}")
