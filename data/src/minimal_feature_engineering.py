import pandas as pd
import numpy as np
def main():
    print("Loading data...")
    train_df = pd.read_csv('./data/processed/train_balanced.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    test_df = pd.read_csv('./data/processed/test.csv')
    print("Engineering features...")
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        # Sort by Time
        df = df.sort_values('Time').reset_index(drop=True)
        # Basic temporal features
        if 'Time_Hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['Time_Hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['Time_Hour'] / 24)
        # Time since last transaction
        df['time_diff'] = df['Time'].diff().fillna(0)
        # Simple rolling mean
        df['amt_rolling_mean'] = df['Amount'].rolling(window=5, min_periods=1).mean()
        # Interaction feature
        df['amt_v1'] = df['Amount'] * df['V1']
        print(f"{name} shape: {df.shape}")
    # Save datasets
    train_df.to_csv('./data/processed/train_enhanced_minimal.csv', index=False)
    val_df.to_csv('./data/processed/val_enhanced_minimal.csv', index=False)
    test_df.to_csv('./data/processed/test_enhanced_minimal.csv', index=False)
    # Save feature list
    feature_cols = [col for col in train_df.columns if col not in ['Class', 'Time']]
    import json
    with open('./models/feature_list_minimal.json', 'w') as f:
        json.dump(feature_cols, f)
    print("Done!")
if __name__ == "__main__":
    main()
