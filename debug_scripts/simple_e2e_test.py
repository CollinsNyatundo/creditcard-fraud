import pandas as pd
import joblib
import time
def simple_test():
    print("Starting simple end-to-end test...")
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
    # Predict
    print("5. Making predictions...")
    start_time = time.perf_counter()
    predictions = model.predict(X_processed)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000 / len(predictions)
    print(f"   Average latency per prediction: {latency_ms:.4f} ms")
    print(f"   Predictions: {predictions}")
    print("   Done.")
    print("Simple end-to-end test completed successfully!")
    return True
if __name__ == "__main__":
    try:
        simple_test()
    except Exception as e:
        print(f"Test failed with error: {e}")
