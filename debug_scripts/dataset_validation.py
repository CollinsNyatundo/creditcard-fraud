import pandas as pd
import numpy as np
import json
import os
def convert_types(obj):
    """Convert non-serializable types to serializable ones"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif pd.api.types.is_extension_array_dtype(obj):
        return str(obj)
    elif hasattr(obj, 'dtype'):
        return str(obj)
    return obj
def validate_dataset(file_path):
    """Validate a single dataset file"""
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        # Get basic info
        info = {
            "file_path": file_path,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }
        # Check for Class column (target variable)
        if 'Class' in df.columns:
            class_distribution = df['Class'].value_counts().to_dict()
            info["class_distribution"] = {int(k): int(v) for k, v in class_distribution.items()}
            info["fraud_percentage"] = float((class_distribution.get(1, 0) / len(df)) * 100) if len(df) > 0 else 0
        # Get statistical summary for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            info["numeric_stats"] = {
                "mean": {k: float(v) for k, v in df[numeric_columns].mean().to_dict().items()},
                "std": {k: float(v) for k, v in df[numeric_columns].std().to_dict().items()},
                "min": {k: float(v) for k, v in df[numeric_columns].min().to_dict().items()},
                "max": {k: float(v) for k, v in df[numeric_columns].max().to_dict().items()}
            }
        return info, None
    except Exception as e:
        return None, str(e)
def validate_all_datasets():
    """Validate all processed datasets"""
    processed_dir = "./data/processed"
    datasets = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    validation_results = {
        "validation_timestamp": pd.Timestamp.now().isoformat(),
        "total_datasets": len(datasets),
        "datasets": {}
    }
    print("Validating processed datasets...\n")
    for dataset in datasets:
        file_path = os.path.join(processed_dir, dataset)
        print(f"Validating {dataset}...")
        info, error = validate_dataset(file_path)
        if error:
            validation_results["datasets"][dataset] = {
                "status": "error",
                "error": error
            }
            print(f"  ERROR: {error}\n")
        else:
            validation_results["datasets"][dataset] = {
                "status": "success",
                "info": info
            }
            print(f"  SUCCESS: {info['rows']} rows, {info['columns']} columns")
            if 'Class' in info.get('column_names', []):
                fraud_pct = info.get('fraud_percentage', 0)
                print(f"  Fraud percentage: {fraud_pct:.4f}%")
            print()
    return validation_results
def compare_datasets(validation_results):
    """Compare enhanced datasets with original processed datasets"""
    print("Comparing dataset schemas...\n")
    # Get base datasets (without enhanced)
    base_datasets = [k for k in validation_results["datasets"].keys()
                     if not k.endswith('_enhanced.csv') and k.endswith('.csv')]
    enhanced_datasets = [k for k in validation_results["datasets"].keys()
                         if k.endswith('_enhanced.csv')]
    comparison_results = {}
    for enhanced in enhanced_datasets:
        base_name = enhanced.replace('_enhanced.csv', '.csv')
        if base_name in validation_results["datasets"]:
            base_info = validation_results["datasets"][base_name]["info"]
            enhanced_info = validation_results["datasets"][enhanced]["info"]
            base_cols = set(base_info["column_names"])
            enhanced_cols = set(enhanced_info["column_names"])
            new_features = list(enhanced_cols - base_cols)
            missing_features = list(base_cols - enhanced_cols)
            comparison_results[enhanced] = {
                "base_dataset": base_name,
                "base_columns": len(base_cols),
                "enhanced_columns": len(enhanced_cols),
                "new_features_count": len(new_features),
                "new_features": new_features[:10],  # Limit to first 10
                "missing_features": missing_features
            }
            print(f"{base_name} -> {enhanced}:")
            print(f"  Base columns: {len(base_cols)}")
            print(f"  Enhanced columns: {len(enhanced_cols)}")
            print(f"  New features added: {len(new_features)}")
            if new_features:
                print(f"    New features: {new_features[:10]}{'...' if len(new_features) > 10 else ''}")
            if missing_features:
                print(f"  Missing features: {missing_features}")
            print()
    return comparison_results
if __name__ == "__main__":
    # Run validation
    results = validate_all_datasets()
    # Compare datasets
    comparison = compare_datasets(results)
    results["dataset_comparisons"] = comparison
    # Save results to JSON
    output_file = "./reports/dataset_validation_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_types)
    print(f"Validation report saved to {output_file}\n")
    print("Dataset validation complete.")
