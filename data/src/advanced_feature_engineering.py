import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def load_and_analyze_data():
    """Load and analyze processed training data"""
    print("Loading processed data...")
    train_df = pd.read_csv('./data/processed/train_balanced.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    test_df = pd.read_csv('./data/processed/test.csv')
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
def apply_feature_selection(train_df, val_df, test_df):
    """Apply feature selection methods"""
    print("Applying feature selection...")
    # For this implementation, we'll keep all engineered features
    # In a production setting, we would implement more sophisticated selection
    # Get all feature columns except Class and Time (which might be ID-like)
    feature_cols = [col for col in train_df.columns if col not in ['Class', 'Time']]
    print(f"Selected {len(feature_cols)} features")
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
