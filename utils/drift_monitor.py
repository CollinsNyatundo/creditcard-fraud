import asyncio
import json
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text
from scipy import special
from scipy.stats import ks_2samp, wasserstein_distance

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

    # ── R16: AUPRC trend tracking ──────────────────────────────────────────────
    # If ground-truth labels are available in the live data, compute AUPRC for
    # this monitoring window. This allows operators to track AUPRC over time.
    auprc_window: dict = {"status": "no_labels_available"}
    if "Class" in live_df.columns and live_df["Class"].sum() > 0:
        try:
            from sklearn.metrics import average_precision_score
            auprc_val = float(average_precision_score(live_df["Class"], live_df["Prediction"]))
            gate_status = "PASS" if auprc_val > 0.70 else "FAIL — REGRESSION DETECTED"
            auprc_window = {
                "status": gate_status,
                "auprc": auprc_val,
                "gate": "> 0.70",
                "window_size": len(live_df),
            }
            print(f"[R16] AUPRC this window: {auprc_val:.4f} [{gate_status}]")
        except Exception as auprc_err:
            auprc_window = {"status": "error", "error": str(auprc_err)}
    else:
        print("[R16] AUPRC trend: no ground-truth labels in live data — skipping AUPRC calculation.")

    report_dict["auprc_window"] = auprc_window

    # ── R17: Amount distribution drift (Wasserstein + KS test) ────────────────
    # Cost function accuracy depends on stable transaction Amount distributions.
    # Track drift to detect if cost estimates are becoming unreliable.
    amount_drift: dict = {"status": "ok"}
    try:
        ref_amounts = ref_df["Amount"].dropna().values
        live_amounts = live_df["Amount"].dropna().values
        if len(ref_amounts) > 0 and len(live_amounts) > 0:
            ks_stat, ks_p = ks_2samp(ref_amounts, live_amounts)
            wass_dist = float(wasserstein_distance(ref_amounts, live_amounts))
            drift_detected = bool(ks_p < 0.05)
            amount_drift = {
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "wasserstein_distance": wass_dist,
                "drift_detected": drift_detected,
                "status": "DRIFT DETECTED — cost estimates may be unreliable" if drift_detected else "stable",
            }
            flag = "[WARN]" if drift_detected else "[OK]"
            print(f"{flag} [R17] Amount distribution: KS={ks_stat:.4f} p={ks_p:.4f} Wasserstein={wass_dist:.2f} | {amount_drift['status']}")
        else:
            amount_drift = {"status": "insufficient_data"}
    except Exception as amt_err:
        amount_drift = {"status": "error", "error": str(amt_err)}

    report_dict["amount_distribution_drift"] = amount_drift

    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[OK] Saved drift report JSON to {json_path}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
