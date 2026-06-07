import os
import pandas as pd
print("Checking created files...")
# Check if enhanced datasets exist
files_to_check = [
    "./data/processed/train_enhanced.csv",
    "./data/processed/val_enhanced.csv",
    "./data/processed/test_enhanced.csv"
]
for file_path in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"[OK] {file_path} exists (size: {size / (1024 * 1024):.1f} MB)")
    else:
        print(f"[FAIL] {file_path} does not exist")
# Check if feature list exists
feature_list_path = "./models/feature_list.json"
if os.path.exists(feature_list_path):
    import json
    with open(feature_list_path, 'r') as f:
        features = json.load(f)
    print(f"[OK] Feature list exists with {len(features)} features")
    print(f"First 10 features: {features[:10]}")
else:
    print(f"[FAIL] {feature_list_path} does not exist")
print("\nChecking data structure...")
try:
    train_df = pd.read_csv("./data/processed/train_enhanced.csv")
    print(f"Training data shape: {train_df.shape}")
    print("Columns:", list(train_df.columns)[:20], "..." if len(train_df.columns) > 20 else "")
except Exception as e:
    print(f"Error reading training data: {e}")
