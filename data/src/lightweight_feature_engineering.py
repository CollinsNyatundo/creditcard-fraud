import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def load_and_sample_data():
    """Load and sample data for faster processing"""
    print("Loading and sampling data...")
    # Load smaller samples for faster processing
    train_df = pd.read_csv('./data/processed/train_balanced.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    test_df = pd.read_csv('./data/processed/test.csv')
    print(f"Original shapes - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    # Sample data for faster processing (taking every 10th row)
    train_df = train_df.iloc[::10].reset_index(drop=True)
    val_df = val_df.iloc[::10].reset_index(drop=True)
    test_df = test_df.iloc[::10].reset_index(drop=True)
    print(f"Sampled shapes - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df
def engineer_basic_features(df, dataset_name):
    """Engineer basic features for faster processing"""
    print(f"Engineering basic features for {dataset_name}...")
    df = df.copy()
    # Sort by Time to ensure proper calculations
    df = df.sort_values('Time').reset_index(drop=True)
    # Basic temporal features
    if 'Time_Hour' in df.columns:
        df['Time_Hour_Sin'] = np.sin(2 * np.pi * df['Time_Hour'] / 24)
        df['Time_Hour_Cos'] = np.cos(2 * np.pi * df['Time_Hour'] / 24)
    # Time since last transaction
    df['time_since_last'] = df['Time'].diff().fillna(0)
    # Simple rolling statistics for Amount (window=5)
    df['amt_mean_5'] = df['Amount'].rolling(window=5, min_periods=1).mean()
    df['amt_std_5'] = df['Amount'].rolling(window=5, min_periods=1).std().fillna(0)
    # Amount deviation
    df['amt_deviation_5'] = df['Amount'] - df['amt_mean_5']
    df['amt_zscore_5'] = np.where(df['amt_std_5'] > 0,
                                  df['amt_deviation_5'] / df['amt_std_5'], 0)
    # Simple interaction features
    df['amt_V1_interaction'] = df['Amount'] * df['V1']
    df['V1_squared'] = df['V1'] ** 2
    # Simple fraud indicators
    overall_mean = df['Amount'].mean()
    overall_std = df['Amount'].std()
    df['amt_overall_zscore'] = (df['Amount'] - overall_mean) / (overall_std + 1e-8)
    # Binary flags for extreme conditions
    amt_95_percentile = df['Amount'].quantile(0.95)
    df['amt_high'] = (df['Amount'] > amt_95_percentile).astype(int)
    return df
def save_datasets(train_df, val_df, test_df):
    """Save enhanced datasets and feature list"""
    # Get common columns
    common_cols = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
    feature_cols = [col for col in common_cols if col not in ['Class', 'Time']]
    print(f"Saving datasets with {len(feature_cols)} features...")
    # Save datasets
    train_df.to_csv('./data/processed/train_enhanced.csv', index=False)
    val_df.to_csv('./data/processed/val_enhanced.csv', index=False)
    test_df.to_csv('./data/processed/test_enhanced.csv', index=False)
    # Save feature list
    import json
    with open('./models/feature_list.json', 'w') as f:
        json.dump(feature_cols, f)
    print("Datasets saved successfully.")
def main():
    """Main function"""
    try:
        # Load and sample data
        train_df, val_df, test_df = load_and_sample_data()
        # Engineer features
        train_enhanced = engineer_basic_features(train_df, "train")
        val_enhanced = engineer_basic_features(val_df, "validation")
        test_enhanced = engineer_basic_features(test_df, "test")
        # Save datasets
        save_datasets(train_enhanced, val_enhanced, test_enhanced)
        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()
