import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
# Create reports directory if it doesn't exist
os.makedirs('./reports', exist_ok=True)
def load_and_validate_data(filepath):
    """Load and perform initial validation of the credit card fraud dataset."""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print("\n--- DATASET LOADING & BASIC INFO ---")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n--- COLUMN INFO ---")
    print(df.info())
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())
    print("\n--- DATA TYPES ---")
    print(df.dtypes)
    return df
def validate_schema(df):
    """Validate dataset schema matches expected structure."""
    print("\n--- SCHEMA VALIDATION ---")
    expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
    missing_columns = set(expected_columns) - set(df.columns)
    extra_columns = set(df.columns) - set(expected_columns)
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        return False
    else:
        print("[OK] All expected columns present")
    if extra_columns:
        print(f"WARNING: Extra columns found: {extra_columns}")
    # Check data types
    numeric_columns = [col for col in df.columns if col not in ['Class']]
    non_numeric = df[numeric_columns].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"ERROR: Non-numeric columns found in numeric fields: {non_numeric}")
        return False
    else:
        print("[OK] All feature columns are numeric")
    # Check Class column
    if df['Class'].dtype != 'int64':
        print(f"WARNING: Class column is not int64 (current type: {df['Class'].dtype})")
    # Check Class values
    class_values = set(df['Class'].unique())
    expected_class_values = {0, 1}
    if class_values != expected_class_values:
        print(f"ERROR: Unexpected Class values: {class_values} (expected: {expected_class_values})")
        return False
    else:
        print("[OK] Class column has correct binary values (0, 1)")
    return True
def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    print("\n--- MISSING VALUES ANALYSIS ---")
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Percentage', ascending=False)
    if missing_df.empty:
        print("[OK] No missing values found in the dataset")
    else:
        print("Missing values found:")
        print(missing_df)
    return missing_df
def analyze_class_distribution(df):
    """Analyze the class distribution for imbalance."""
    print("\n--- CLASS DISTRIBUTION ANALYSIS ---")
    class_counts = df['Class'].value_counts()
    class_percent = df['Class'].value_counts(normalize=True) * 100
    print("Class counts:")
    for class_val, count in class_counts.items():
        percent = class_percent[class_val]
        print(f"  Class {class_val}: {count} ({percent:.4f}%)")
    # Calculate imbalance ratio
    fraud_count = class_counts.get(1, 0)
    non_fraud_count = class_counts.get(0, 0)
    if fraud_count > 0:
        imbalance_ratio = non_fraud_count / fraud_count
        print(f"\nImbalance ratio (non-fraud:fraud): {imbalance_ratio:.0f}:1")
        print(f"Fraud rate: {class_percent.get(1, 0):.4f}%")
    return class_counts, class_percent
def generate_summary_statistics(df):
    """Generate summary statistics for all features."""
    print("\n--- SUMMARY STATISTICS ---")
    # Separate numerical and categorical features (though all should be numerical)
    numerical_features = [col for col in df.columns if col not in ['Class']]
    # Summary statistics for all numerical features
    stats = df[numerical_features].describe()
    print("Numerical features statistics:")
    print(stats)
    # Summary statistics for Class
    print("\nClass statistics:")
    print(df['Class'].describe())
    return stats
def create_visualizations(df):
    """Create visualizations for EDA report."""
    print("\n--- CREATING VISUALIZATIONS ---")
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Credit Card Fraud Detection - Exploratory Data Analysis', fontsize=16)
    # 1. Class distribution
    class_counts = df['Class'].value_counts()
    axes[0, 0].bar(class_counts.index, class_counts.values, color=['skyblue', 'red'])
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_xlabel('Class (0: Non-Fraud, 1: Fraud)')
    axes[0, 0].set_ylabel('Count')
    # Add value labels on bars
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v, str(v), ha='center', va='bottom')
    # 2. Amount distribution by class (log scale)
    df_non_fraud = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]
    axes[0, 1].hist(df_non_fraud['Amount'], bins=50, alpha=0.7, label='Non-Fraud', color='skyblue')
    axes[0, 1].hist(df_fraud['Amount'], bins=50, alpha=0.7, label='Fraud', color='red')
    axes[0, 1].set_title('Transaction Amount Distribution by Class')
    axes[0, 1].set_xlabel('Amount')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    # 3. Time distribution by class
    axes[1, 0].hist(df_non_fraud['Time'], bins=50, alpha=0.7, label='Non-Fraud', color='skyblue')
    axes[1, 0].hist(df_fraud['Time'], bins=50, alpha=0.7, label='Fraud', color='red')
    axes[1, 0].set_title('Transaction Time Distribution by Class')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    # 4. Fraud rate over time (binned)
    # Create time bins
    df['TimeBin'] = pd.cut(df['Time'], bins=50)
    fraud_rate_by_time = df.groupby('TimeBin')['Class'].mean()
    axes[1, 1].plot(range(len(fraud_rate_by_time)), fraud_rate_by_time.values, marker='o', linewidth=2)
    axes[1, 1].set_title('Fraud Rate Over Time')
    axes[1, 1].set_xlabel('Time Bins')
    axes[1, 1].set_ylabel('Fraud Rate')
    axes[1, 1].set_ylim(0, max(fraud_rate_by_time.values) * 1.1)
    plt.tight_layout()
    plt.savefig('./reports/eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualizations saved to ./reports/eda_visualizations.png")
def generate_html_report(df, schema_valid, missing_df, class_counts, class_percent, stats):
    """Generate HTML report for EDA."""
    print("\n--- GENERATING HTML REPORT ---")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Card Fraud Detection - EDA Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
            .section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>Credit Card Fraud Detection - Exploratory Data Analysis Report</h1>
        <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="section">
            <h2>Dataset Overview</h2>
            <p><strong>Shape:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns</p>
            <p><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>
        </div>
        <div class="section">
            <h2>Schema Validation</h2>
            <p class="{'success' if schema_valid else 'error'}">
                {'[OK] Schema validation passed' if schema_valid else '[FAIL] Schema validation failed'}
            </p>
        </div>
        <div class="section">
            <h2>Missing Values Analysis</h2>
    """
    if missing_df.empty:
        html_content += "<p class='success'>[OK] No missing values found in the dataset</p>"
    else:
        html_content += """
            <table>
                <tr><th>Column</th><th>Missing Count</th><th>Missing Percentage</th></tr>
        """
        for idx, row in missing_df.iterrows():
            html_content += f"<tr><td>{idx}</td><td>{row['Missing Count']}</td><td>{row['Missing Percentage']:.4f}%</td></tr>"
        html_content += "</table>"
    html_content += f"""
        </div>
        <div class="section">
            <h2>Class Distribution</h2>
            <table>
                <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
    """
    for class_val, count in class_counts.items():
        percent = class_percent[class_val]
        html_content += f"<tr><td>{class_val}</td><td>{count:,}</td><td>{percent:.4f}%</td></tr>"
    fraud_count = class_counts.get(1, 0)
    non_fraud_count = class_counts.get(0, 0)
    if fraud_count > 0:
        imbalance_ratio = non_fraud_count / fraud_count
        html_content += f"""
            </table>
            <p><strong>Imbalance Ratio (non-fraud:fraud):</strong> {imbalance_ratio:.0f}:1</p>
            <p><strong>Fraud Rate:</strong> {class_percent.get(1, 0):.4f}%</p>
        """
    html_content += """
        </div>
        <div class="section">
            <h2>Summary Statistics - Numerical Features</h2>
            <table>
                <tr><th>Statistic</th>
    """
    # Add column headers
    numerical_features = [col for col in df.columns if col not in ['Class']]
    for col in numerical_features[:10]:  # First 10 columns
        html_content += f"<th>{col}</th>"
    html_content += "</tr>"
    # Add statistics rows
    for stat in stats.index:
        html_content += f"<tr><td>{stat}</td>"
        for col in numerical_features[:10]:
            html_content += f"<td>{stats.loc[stat, col]:.4f}</td>"
        html_content += "</tr>"
    html_content += """
            </table>
            <p><em>Note: Only first 10 features shown for brevity. See CSV file for complete statistics.</em></p>
        </div>
        <div class="section">
            <h2>Visualizations</h2>
            <p>Key visualizations have been saved to <code>./reports/eda_visualizations.png</code></p>
            <img src="eda_visualizations.png" alt="EDA Visualizations" style="max-width:100%; height:auto;">
        </div>
        <div class="section">
            <h2>Data Quality Summary</h2>
    """
    if schema_valid and missing_df.empty:
        html_content += "<p class='success'>[OK] Data quality is high - no schema issues or missing values detected</p>"
    else:
        html_content += "<p class='warning'>⚠ Data quality issues detected - see details above</p>"
    html_content += """
        </div>
    </body>
    </html>
    """
    with open('./reports/eda_report.html', 'w') as f:
        f.write(html_content)
    # Also save summary statistics to CSV for detailed analysis
    stats.to_csv('./reports/feature_statistics.csv')
    print("HTML report saved to ./reports/eda_report.html")
    print("Feature statistics saved to ./reports/feature_statistics.csv")
def main():
    """Main function to run the complete EDA pipeline."""
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    # Load and validate data
    df = load_and_validate_data('./data/raw/creditcard.csv')
    # Validate schema
    schema_valid = validate_schema(df)
    # Analyze missing values
    missing_df = analyze_missing_values(df)
    # Analyze class distribution
    class_counts, class_percent = analyze_class_distribution(df)
    # Generate summary statistics
    stats = generate_summary_statistics(df)
    # Create visualizations
    create_visualizations(df)
    # Generate HTML report
    generate_html_report(df, schema_valid, missing_df, class_counts, class_percent, stats)
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS COMPLETE")
    print("=" * 60)
if __name__ == "__main__":
    main()
