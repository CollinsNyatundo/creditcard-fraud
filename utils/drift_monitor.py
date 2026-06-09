import asyncio
import json
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text
from scipy import special

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.engine import AsyncSessionLocal
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

async def fetch_live_data():
    """Fetch recent live predictions from PostgreSQL."""
    print("Fetching live predictions from PostgreSQL database...")
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("""
                    SELECT amount, fraud_probability 
                    FROM predictions 
                    ORDER BY created_at DESC 
                    LIMIT 10000
                """)
            )
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=["Amount", "Prediction"])
            print(f"Fetched {len(df)} live predictions.")
            return df
    except Exception as e:
        print(f"Warning: Failed to fetch live predictions from database: {e}")
        return pd.DataFrame(columns=["Amount", "Prediction"])

def load_reference_data():
    """Load baseline training data and generate reference predictions."""
    print("Loading baseline training reference data...")
    train_path = "./data/processed/train_enhanced.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training reference data not found at {train_path}")
    
    train_df = pd.read_csv(train_path)
    # Sample reference data for stability and speed
    sample_df = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    
    # Load model to get baseline predictions
    model_path = "./models/optimized_lightgbm.pkl"
    feature_list_path = "./models/feature_list.json"
    if not os.path.exists(model_path) or not os.path.exists(feature_list_path):
        raise FileNotFoundError("Model or feature list not found in ./models. Run training first.")
        
    import joblib
    model = joblib.load(model_path)
    with open(feature_list_path, "r") as f:
        feature_cols = json.load(f)
        
    X_ref = sample_df[feature_cols]
    
    # Predict raw scores
    raw_preds = model.predict(X_ref)
    
    # Check for Focal Loss config
    init_score = 0.0
    is_focal_loss = False
    threshold_path = "./models/optimal_threshold_v2.json"
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, "r") as f:
                config_data = json.load(f)
                init_score = config_data.get("init_score", 0.0)
                is_focal_loss = config_data.get("is_focal_loss", False)
        except Exception:
            pass
            
    # Apply transformation
    if is_focal_loss:
        ref_probs = special.expit(raw_preds + init_score)
    else:
        ref_probs = raw_preds
        
    ref_df = pd.DataFrame({
        "Amount": sample_df["Amount"].values,
        "Prediction": ref_probs
    })
    return ref_df

async def main():
    print("=" * 60)
    print("PRODUCTION DATA DRIFT MONITORING WORKER")
    print("=" * 60)
    
    try:
        ref_df = load_reference_data()
    except Exception as e:
        print(f"[FAIL] Error loading reference data: {e}")
        return
        
    live_df = await fetch_live_data()
    
    # If the database is empty or has too few entries, simulate mock live data for validation
    if len(live_df) < 10:
        print("Database predictions ledger is empty or has too few entries (<10).")
        print("Generating mock live data for testing/validation...")
        np.random.seed(42)
        live_df = ref_df.copy()
        # Perturb Amount slightly
        live_df["Amount"] = live_df["Amount"] * np.random.uniform(0.92, 1.08, size=len(live_df))
        # Perturb Predictions slightly
        live_df["Prediction"] = np.clip(
            live_df["Prediction"] + np.random.normal(0, 0.03, size=len(live_df)),
            0.0, 1.0
        )
        
    print("Running Evidently Data Drift analysis...")
    # Setup evidently report
    report = Report(metrics=[
        DataDriftPreset(columns=["Amount", "Prediction"])
    ])
    
    report.run(reference_data=ref_df, current_data=live_df)
    
    # Ensure directories exist
    os.makedirs("./reports", exist_ok=True)
    
    # Save dashboard HTML
    html_path = "./reports/drift_dashboard.html"
    report.save_html(html_path)
    print(f"[OK] Saved interactive drift dashboard to {html_path}")
    
    # Save report JSON
    json_path = "./reports/drift_report.json"
    report_dict = report.as_dict()
    
    # Add custom metadata versioning tags required for reports
    report_dict["schema_version"] = 1
    report_dict["timestamp"] = pd.Timestamp.now().isoformat()
    report_dict["generated_at"] = pd.Timestamp.now().isoformat()
    report_dict["source_script"] = "utils/drift_monitor.py"
    report_dict["artifact_of"] = "creditcard-fraud-pipeline"
    
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[OK] Saved drift report JSON to {json_path}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
