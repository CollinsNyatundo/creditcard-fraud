import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
def load_train_data(filepath):
    """Load the training dataset."""
    print("Loading training dataset...")
    df = pd.read_csv(filepath)
    print(f"Training dataset loaded with shape: {df.shape}")
    return df
def analyze_class_distribution(df, name):
    """Analyze the class distribution."""
    print(f"\n--- {name} CLASS DISTRIBUTION ---")
    class_counts = df['Class'].value_counts()
    class_percent = df['Class'].value_counts(normalize=True) * 100
    for class_val, count in class_counts.items():
        percent = class_percent[class_val]
        print(f"  Class {class_val}: {count} ({percent:.4f}%)")
    # Calculate imbalance ratio
    fraud_count = class_counts.get(1, 0)
    non_fraud_count = class_counts.get(0, 0)
    if fraud_count > 0:
        imbalance_ratio = non_fraud_count / fraud_count
        print(f"  Imbalance ratio (non-fraud:fraud): {imbalance_ratio:.0f}:1")
    return class_counts, class_percent
def handle_class_imbalance(df, random_state=42):
    """
    Handle class imbalance using a hybrid approach:
    1. Apply SMOTE to oversample the minority class
    2. Apply strategic undersampling to the majority class
    Target balanced ratio: approximately 1:5 (fraud to non-fraud)
    """
    print("\n--- HANDLING CLASS IMBALANCE ---")
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    print(f"Before sampling - X shape: {X.shape}, y shape: {y.shape}")
    print(f"Original class distribution: {y.value_counts().to_dict()}")
    # Identify categorical columns (object dtype)
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {len(numerical_features)}")
    # Create preprocessor to handle categorical features
    if categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
        # Transform the features
        X_transformed = preprocessor.fit_transform(X)
        # Get feature names after transformation
        numerical_feature_names = numerical_features
        categorical_feature_names = []
        if categorical_features:
            categorical_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        feature_names = numerical_feature_names + categorical_feature_names
        # Create DataFrame with transformed features
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        print(f"Transformed features shape: {X_transformed_df.shape}")
    else:
        X_transformed_df = X
        preprocessor = None
    # Apply SMOTE with a more conservative sampling strategy
    # Target a 1:10 ratio (fraud to non-fraud)
    smote = SMOTE(sampling_strategy=0.1, random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_transformed_df, y)
    print(f"After SMOTE - X shape: {X_resampled.shape}, y shape: {y_resampled.shape}")
    print(f"SMOTE class distribution: {y_resampled.value_counts().to_dict()}")
    # Apply undersampling to further balance the dataset
    # Target a 1:5 ratio (fraud to non-fraud)
    undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=random_state)
    X_final, y_final = undersampler.fit_resample(X_resampled, y_resampled)
    print(f"After undersampling - X shape: {X_final.shape}, y shape: {y_final.shape}")
    print(f"Final class distribution: {y_final.value_counts().to_dict()}")
    # Create final dataframe
    df_final = pd.DataFrame(X_final, columns=X_resampled.columns)
    df_final['Class'] = y_final
    return df_final, preprocessor
def save_balanced_train(df_balanced, preprocessor, output_path, preprocessor_path):
    """Save the balanced training dataset and preprocessor."""
    print(f"\n--- SAVING BALANCED TRAINING SET ---")
    df_balanced.to_csv(output_path, index=False)
    print(f"Balanced training set saved to: {output_path}")
    # Save preprocessor if it exists
    if preprocessor is not None:
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")
def verify_no_changes_to_val_test(val_before, test_before, val_path, test_path):
    """Verify that validation and test sets are unchanged after imbalance handling.

    Args:
        val_before:  DataFrame snapshot of val.csv taken *before* any processing.
        test_before: DataFrame snapshot of test.csv taken *before* any processing.
        val_path:    Path to the val.csv written after processing.
        test_path:   Path to the test.csv written after processing.
    """
    print("\n--- VERIFYING NO CHANGES TO VALIDATION AND TEST SETS ---")

    # Load current (post-processing) validation and test sets
    current_val = pd.read_csv(val_path)
    current_test = pd.read_csv(test_path)

    # Compare against the snapshots captured *before* imbalance handling
    val_identical = current_val.equals(val_before)
    test_identical = current_test.equals(test_before)

    if val_identical:
        print("[OK] Validation set is unchanged")
    else:
        print("[FAIL] Validation set has been modified")
    if test_identical:
        print("[OK] Test set is unchanged")
    else:
        print("[FAIL] Test set has been modified")

    return val_identical and test_identical
def main():
    """Main function to handle class imbalance in the training set."""
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION - CLASS IMBALANCE HANDLING")
    print("=" * 70)

    # Define file paths
    original_train_path = './data/processed/train.csv'
    balanced_train_path = './data/processed/train_balanced.csv'
    preprocessor_path   = './models/balancing_preprocessor.pkl'
    val_path  = './data/processed/val.csv'
    test_path = './data/processed/test.csv'

    # ── Snapshot val/test BEFORE any processing ──────────────────────────
    # This lets us verify that imbalance handling only touched train.csv.
    val_before  = pd.read_csv(val_path)
    test_before = pd.read_csv(test_path)

    # Load training data
    train_df = load_train_data(original_train_path)

    # Analyze original class distribution
    analyze_class_distribution(train_df, "ORIGINAL TRAINING SET")

    # Handle class imbalance
    balanced_train_df, preprocessor = handle_class_imbalance(train_df, random_state=42)

    # Analyze balanced class distribution
    analyze_class_distribution(balanced_train_df, "BALANCED TRAINING SET")

    # Save balanced training set
    save_balanced_train(
        balanced_train_df, preprocessor, balanced_train_path, preprocessor_path
    )

    # Verify that validation and test sets are unchanged (using pre-processing snapshots)
    verify_no_changes_to_val_test(val_before, test_before, val_path, test_path)

    print("\n" + "=" * 70)
    print("CLASS IMBALANCE HANDLING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
