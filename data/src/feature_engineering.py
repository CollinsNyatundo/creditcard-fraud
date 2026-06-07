import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import joblib
# Create processed data directory if it doesn't exist
os.makedirs('./data/processed', exist_ok=True)
def load_data(filepath):
    """Load the credit card fraud dataset."""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with shape: {df.shape}")
    return df
def engineer_features(df):
    """Perform feature engineering on the dataset."""
    print("\n--- FEATURE ENGINEERING ---")
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    # Temporal features from Time column
    # Hours since first transaction (assuming Time is in seconds)
    df_processed['Time_Hours'] = df_processed['Time'] / 3600
    # Normalized time
    df_processed['Time_Normalized'] = (df_processed['Time'] - df_processed['Time'].mean()) / df_processed['Time'].std()
    # Cyclical encoding of time (hours in day)
    df_processed['Time_Hour'] = (df_processed['Time'] / 3600) % 24
    df_processed['Time_Hour_Sin'] = np.sin(2 * np.pi * df_processed['Time_Hour'] / 24)
    df_processed['Time_Hour_Cos'] = np.cos(2 * np.pi * df_processed['Time_Hour'] / 24)
    # Amount features
    # Log transformation (adding 1 to handle zero values)
    df_processed['Amount_Log'] = np.log1p(df_processed['Amount'])
    # Normalized amount
    df_processed['Amount_Normalized'] = (df_processed['Amount'] - df_processed['Amount'].mean()) / df_processed['Amount'].std()
    # Amount bins (low/medium/high)
    df_processed['Amount_Bin'] = pd.cut(df_processed['Amount'],
                                        bins=[0, 10, 100, np.inf],
                                        labels=['Low', 'Medium', 'High'])
    # One-hot encode amount bins
    amount_bin_dummies = pd.get_dummies(df_processed['Amount_Bin'], prefix='Amount')
    df_processed = pd.concat([df_processed, amount_bin_dummies], axis=1)
    print(f"Feature engineering completed. New shape: {df_processed.shape}")
    return df_processed
def scale_features(df):
    """Apply robust scaling to V1-V28 and Amount features."""
    print("\n--- FEATURE SCALING ---")
    # Identify features to scale
    features_to_scale = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                         'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                         'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                         'Amount', 'Amount_Log', 'Amount_Normalized']
    # Apply robust scaling
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    print(f"Feature scaling completed. Scaled features: {len(features_to_scale)}")
    return df_scaled, scaler, features_to_scale
def create_temporal_splits(df, train_size=0.70, val_size=0.15, random_state=42):
    """
    Create strictly chronological train/validation/test splits using hard
    time-boundary cuts.

    The data is sorted by the 'Time' column and sliced by index position so
    that:
      - train  = first 70 % of transactions by time
      - val    = next  15 % of transactions by time
      - test   = last  15 % of transactions by time

    No shuffling or stratified sampling is applied, which prevents any form
    of temporal data leakage (future transactions appearing in the training
    set).

    Args:
        df:          Preprocessed DataFrame containing a 'Time' column.
        train_size:  Fraction of data for training (default 0.70).
        val_size:    Fraction of data for validation (default 0.15).
        random_state: Kept for API compatibility; not used in slicing.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("\n--- CREATING TEMPORAL SPLITS (hard chronological cutoff) ---")

    # Sort by Time to maintain strict temporal ordering
    df_sorted = df.sort_values('Time').reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_size)
    val_end   = train_end + int(n * val_size)

    train_df = df_sorted.iloc[:train_end].copy()
    val_df   = df_sorted.iloc[train_end:val_end].copy()
    test_df  = df_sorted.iloc[val_end:].copy()

    print(f"Train set shape:      {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape:       {test_df.shape}")

    # ── Temporal integrity assertion ────────────────────────────────────
    # These must hold for a truly non-leaky split:
    assert train_df['Time'].max() <= val_df['Time'].min(), (
        "LEAKAGE DETECTED: training set contains transactions later than "
        "the earliest validation transaction."
    )
    assert val_df['Time'].max() <= test_df['Time'].min(), (
        "LEAKAGE DETECTED: validation set contains transactions later than "
        "the earliest test transaction."
    )
    print("[OK] Temporal integrity verified: "
          f"max(train.Time)={train_df['Time'].max():.0f} "
          f"<= min(val.Time)={val_df['Time'].min():.0f} "
          f"<= min(test.Time)={test_df['Time'].min():.0f}")

    return train_df, val_df, test_df
def analyze_splits(train_df, val_df, test_df):
    """Analyze the class distribution in each split."""
    print("\n--- SPLIT ANALYSIS ---")
    splits = [
        ('Train', train_df),
        ('Validation', val_df),
        ('Test', test_df)
    ]
    for name, df in splits:
        class_counts = df['Class'].value_counts()
        class_percent = df['Class'].value_counts(normalize=True) * 100
        print(f"\n{name} Set Class Distribution:")
        for class_val, count in class_counts.items():
            percent = class_percent[class_val]
            print(f"  Class {class_val}: {count} ({percent:.4f}%)")
        # Calculate imbalance ratio
        fraud_count = class_counts.get(1, 0)
        non_fraud_count = class_counts.get(0, 0)
        if fraud_count > 0:
            imbalance_ratio = non_fraud_count / fraud_count
            print(f"  Imbalance ratio (non-fraud:fraud): {imbalance_ratio:.0f}:1")
    return True
def save_splits(train_df, val_df, test_df, scaler, features_scaled):
    """Save the processed splits and preprocessing artifacts."""
    print("\n--- SAVING SPLITS AND ARTIFACTS ---")
    # Save splits
    train_df.to_csv('./data/processed/train.csv', index=False)
    val_df.to_csv('./data/processed/val.csv', index=False)
    test_df.to_csv('./data/processed/test.csv', index=False)
    joblib.dump(scaler, './models/preprocessor.pkl')
    # Save features scaled list
    with open('./models/features_scaled.txt', 'w') as f:
        for feature in features_scaled:
            f.write(f"{feature}\n")
    print("Splits and preprocessing artifacts saved successfully.")
def main():
    """Main function to run the complete feature engineering and preprocessing pipeline."""
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION - FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 70)
    # Load data
    df = load_data('./data/raw/creditcard.csv')
    # Engineer features
    df_engineered = engineer_features(df)
    # Scale features
    df_scaled, scaler, features_scaled = scale_features(df_engineered)
    # Create temporal splits
    train_df, val_df, test_df = create_temporal_splits(df_scaled, test_size=0.15, val_size=0.15, random_state=42)
    # Analyze splits
    analyze_splits(train_df, val_df, test_df)
    # Save splits and artifacts
    save_splits(train_df, val_df, test_df, scaler, features_scaled)
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING & PREPROCESSING COMPLETE")
    print("=" * 70)
if __name__ == "__main__":
    main()
