import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def encode_amount_bin(df):
    df = df.copy()
    if 'Amount_Bin' in df.columns:
        df['Amount_Bin_Low'] = (df['Amount_Bin'] == 'Low').astype(float)
        df['Amount_Bin_Medium'] = (df['Amount_Bin'] == 'Medium').astype(float)
        df['Amount_Bin_High'] = (df['Amount_Bin'] == 'High').astype(float)
        df['Amount_Bin_nan'] = (df['Amount_Bin'] == 'nan').astype(float)
        df = df.drop('Amount_Bin', axis=1)
    return df

def load_and_analyze_data():
    """Load and analyze processed training data"""
    print("Loading processed data...")
    train_df = pd.read_csv('./data/processed/train_balanced.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    test_df = pd.read_csv('./data/processed/test.csv')
    
    # One-hot encode Amount_Bin for validation and test splits
    val_df = encode_amount_bin(val_df)
    test_df = encode_amount_bin(test_df)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    # Check class distribution
    print(f"Train class distribution:\n{train_df['Class'].value_counts(normalize=True)}")
    print(f"Validation class distribution:\n{val_df['Class'].value_counts(normalize=True)}")
    return train_df, val_df, test_df
def engineer_temporal_features(df):
    """Create temporal pattern features"""
    print("Engineering temporal features...")
    df = df.copy()
    # Cyclical encoding for hour (assuming Time_hour exists)
    if 'Time_hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['Time_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Time_hour'] / 24)
    # Time since last transaction (requires sorting by Time)
    df = df.sort_values('Time').reset_index(drop=True)
    df['time_since_last'] = df['Time'].diff().fillna(0)
    # Time of day categories
    if 'Time_hour' in df.columns:
        df['is_night'] = ((df['Time_hour'] >= 22) | (df['Time_hour'] <= 6)).astype(int)
        df['is_weekend'] = (df['Time_day'] >= 5).astype(int) if 'Time_day' in df.columns else 0
    return df
def engineer_statistical_features(df):
    """Create statistical aggregation features"""
    print("Engineering statistical features...")
    df = df.copy()
    # Sort by Time to ensure proper rolling calculations
    df = df.sort_values('Time').reset_index(drop=True)
    # Rolling statistics for Amount
    for window in [3, 5, 10]:
        df[f'amt_mean_{window}'] = df['Amount'].rolling(window=window, min_periods=1).mean()
        df[f'amt_std_{window}'] = df['Amount'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'amt_min_{window}'] = df['Amount'].rolling(window=window, min_periods=1).min()
        df[f'amt_max_{window}'] = df['Amount'].rolling(window=window, min_periods=1).max()
        # Amount deviation features
        df[f'amt_deviation_{window}'] = df['Amount'] - df[f'amt_mean_{window}']
        df[f'amt_zscore_{window}'] = np.where(df[f'amt_std_{window}'] > 0,
                                              df[f'amt_deviation_{window}'] / df[f'amt_std_{window}'], 0)
    # Transaction velocity features
    df['tx_count_10'] = df['Time'].rolling(window=10, min_periods=1).count()
    df['avg_time_diff'] = df['time_since_last'].rolling(window=10, min_periods=1).mean()
    return df
def engineer_interaction_features(df):
    """Generate interaction and polynomial features"""
    print("Engineering interaction features...")
    df = df.copy()
    # Top V-features from baseline model (based on typical feature importance)
    # In a real scenario, we would load feature importance from the baseline model
    top_v_features = ['V1', 'V4', 'V7', 'V11', 'V12']  # Placeholder values
    # Amount × V-feature interactions
    for v_feat in top_v_features:
        if v_feat in df.columns:
            df[f'amt_{v_feat}_interaction'] = df['Amount'] * df[v_feat]
    # Polynomial features for top predictive V-columns
    poly_v_features = ['V1', 'V4', 'V11']  # Placeholder values
    for v_feat in poly_v_features:
        if v_feat in df.columns:
            df[f'{v_feat}_squared'] = df[v_feat] ** 2
    # Amount ratio features
    df['amt_ratio_3'] = df['Amount'] / (df['amt_mean_3'] + 1e-8)  # Add small epsilon to avoid division by zero
    return df
def engineer_fraud_specific_features(df):
    """Implement fraud-specific anomaly features"""
    print("Engineering fraud-specific features...")
    df = df.copy()
    # Sort by Time
    df = df.sort_values('Time').reset_index(drop=True)
    # Amount deviation from overall distribution
    overall_mean = df['Amount'].mean()
    overall_std = df['Amount'].std()
    df['amt_overall_zscore'] = (df['Amount'] - overall_mean) / (overall_std + 1e-8)
    # Expanding window statistics (user's typical spending)
    df['amt_cumsum'] = df['Amount'].expanding(min_periods=1).sum()
    df['amt_cumcount'] = df['Amount'].expanding(min_periods=1).count()
    df['amt_cummean'] = df['amt_cumsum'] / df['amt_cumcount']
    # Binary flags for extreme conditions
    amt_99_percentile = df['Amount'].quantile(0.99)
    amt_1_percentile = df['Amount'].quantile(0.01)
    df['amt_extremely_high'] = (df['Amount'] > amt_99_percentile).astype(int)
    df['amt_extremely_low'] = (df['Amount'] < amt_1_percentile).astype(int)
    return df
def add_clustering_features(train_df, val_df, test_df, base_features):
    """Dynamically determine optimal clusters on train set, fit KMeans, and append cluster one-hot encodings."""
    print("Dynamically fitting KMeans clusterer on train set...")
    import joblib
    import os
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # 1. Select numeric features
    train_numeric = train_df[base_features].select_dtypes(include=[np.number])
    val_numeric = val_df[base_features].select_dtypes(include=[np.number])
    test_numeric = test_df[base_features].select_dtypes(include=[np.number])
    
    # 2. Find optimal K via silhouette analysis
    sample_size = min(10000, len(train_numeric))
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_numeric), sample_size, replace=False)
    X_sample = train_numeric.iloc[sample_indices]
    
    best_k = 3
    best_score = -1.0
    for k in range(3, 7): # range 3 to 6 is sufficient and fast
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        print(f"  Dynamic K search - K={k}: Silhouette Score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Optimal cluster size: K={best_k}")
    
    # 3. Fit final KMeans strictly on training set
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_final.fit(train_numeric)
    
    # Save the fitted clusterer
    os.makedirs("./models", exist_ok=True)
    joblib.dump(kmeans_final, "./models/behavior_clusterer.pkl")
    print("Serialized KMeans behavior clusterer to ./models/behavior_clusterer.pkl")
    
    # Save config
    with open("./models/behavior_clusterer_config.json", "w") as f:
        import json
        json.dump({
            "optimal_k": best_k,
            "feature_names": list(train_numeric.columns)
        }, f, indent=2)
        
    # 4. Predict clusters
    train_clusters = kmeans_final.predict(train_numeric)
    val_clusters = kmeans_final.predict(val_numeric)
    test_clusters = kmeans_final.predict(test_numeric)
    
    # 5. Append one-hot encoded columns
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    for i in range(best_k):
        col_name = f"cluster_{i}"
        train_df[col_name] = (train_clusters == i).astype(float)
        val_df[col_name] = (val_clusters == i).astype(float)
        test_df[col_name] = (test_clusters == i).astype(float)
        
    print(f"Appended one-hot cluster features cluster_0 to cluster_{best_k-1}")
    return train_df, val_df, test_df

def apply_feature_selection(train_df, val_df, test_df):
    """Apply feature selection methods"""
    print("Applying feature selection...")
    # Get all feature columns except Class and Time
    feature_cols = [col for col in train_df.columns if col not in ['Class', 'Time']]
    
    # Apply dynamic KMeans clustering features to avoid lookahead leakage
    train_df, val_df, test_df = add_clustering_features(train_df, val_df, test_df, feature_cols)
    
    # Re-calculate feature list including the new cluster columns
    feature_cols = [col for col in train_df.columns if col not in ['Class', 'Time']]
    print(f"Selected {len(feature_cols)} features (including clustering encodings)")
    return train_df, val_df, test_df, feature_cols

def main():
    """Main function to run feature engineering pipeline"""
    # Load data
    train_df, val_df, test_df = load_and_analyze_data()
    # Apply feature engineering steps
    train_df = engineer_temporal_features(train_df)
    train_df = engineer_statistical_features(train_df)
    train_df = engineer_interaction_features(train_df)
    train_df = engineer_fraud_specific_features(train_df)
    val_df = engineer_temporal_features(val_df)
    val_df = engineer_statistical_features(val_df)
    val_df = engineer_interaction_features(val_df)
    val_df = engineer_fraud_specific_features(val_df)
    test_df = engineer_temporal_features(test_df)
    test_df = engineer_statistical_features(test_df)
    test_df = engineer_interaction_features(test_df)
    test_df = engineer_fraud_specific_features(test_df)
    # Apply feature selection
    train_df, val_df, test_df, feature_cols = apply_feature_selection(train_df, val_df, test_df)
    # Save enhanced datasets
    print("Saving enhanced datasets...")
    train_df.to_csv('./data/processed/train_enhanced.csv', index=False)
    val_df.to_csv('./data/processed/val_enhanced.csv', index=False)
    test_df.to_csv('./data/processed/test_enhanced.csv', index=False)
    # Save feature list
    import json
    with open('./models/feature_list.json', 'w') as f:
        json.dump(feature_cols, f)
    print(f"Feature engineering completed. Engineered {len(feature_cols)} features.")
    print("Enhanced datasets saved successfully.")
if __name__ == "__main__":
    main()
