# Pipeline Integrity & Tech Debt Remediation — Implementation Plan

> **For Agent:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Eliminate pipeline race conditions/silent overwrites, improve latency benchmarking accuracy, clean up duplicate files, and implement proper coverage gating and integration tests.

**Architecture:**
1. Rename feature engineering outputs for lightweight and minimal modules to avoid overwriting advanced feature inputs.
2. Standardize latency benchmarks to measure NumPy-backed inference and use high-resolution `time.perf_counter()`.
3. Purge duplicate files (raw data duplicate, redundant E2E scripts).
4. Add comprehensive integration testing using real model artifacts.
5. Create a legal data license file for the CC BY-NC-SA 4.0 raw dataset.

**Tech Stack:** Python 3.10+ · pytest · pytest-cov · NumPy · Pandas

**Design Decisions / Audit Reference:**
- Production Audit Report: P1-2, P1-4, P2-1, P2-2, P2-4, P2-5, P2-6
- Claims Validity Matrix: Item 11, Item 3, Item 10

---

## Mandatory Gate

> ⚠️ **Do not begin this remediation plan until the implementation plan itself is approved by the user.**
> ⚠️ **All tasks must end with a single semantic git commit. Do not squash multiple tasks into one commit.**

---

### Task 1: Rename Lightweight & Minimal Feature Engineering Outputs

**Files:**
- Modify: [`data/src/lightweight_feature_engineering.py`](file:///d:/Projects/ai-ml/creditcard-fraud/data/src/lightweight_feature_engineering.py)
- Modify: [`data/src/minimal_feature_engineering.py`](file:///d:/Projects/ai-ml/creditcard-fraud/data/src/minimal_feature_engineering.py)

**Step 1: Update Output Filenames**
Change the output file paths in the lightweight feature engineering script from:
`train_enhanced.csv`, `val_enhanced.csv`, `test_enhanced.csv`
to:
`train_enhanced_lite.csv`, `val_enhanced_lite.csv`, `test_enhanced_lite.csv`.

And in the minimal feature engineering script to:
`train_enhanced_minimal.csv`, `val_enhanced_minimal.csv`, `test_enhanced_minimal.csv`.

**Step 2: Run Both Scripts**
Execute the scripts to ensure they write to the new paths without overwriting the advanced outputs:
```bash
python data/src/lightweight_feature_engineering.py
python data/src/minimal_feature_engineering.py
```
Check that the `data/processed/` directory contains all variants simultaneously.

**Step 3: Commit**
```bash
git add data/src/lightweight_feature_engineering.py data/src/minimal_feature_engineering.py
git commit -m "fix: redirect lightweight and minimal feature engineering output paths to prevent data corruption"
```

---

### Task 2: Standardize Latency Benchmark Methodology

**Files:**
- Modify: [`model/src/hyperparameter_tuning.py`](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/hyperparameter_tuning.py) (or `hyperparameter_tuning_fixed.py` if not yet consolidated)

**Step 1: Replace time.time() with time.perf_counter()**
Locate the latency benchmarking block in the tuning script. Change it to use `time.perf_counter()` to obtain higher precision.

**Step 2: Feed NumPy Arrays to Predict**
Instead of using `.iterrows()`, convert the evaluation samples to a NumPy array or list of lists before running the prediction loops to bypass Pandas overhead:
```python
# Convert to numpy values to isolate prediction speed
sample_array = sample_X.values

start_time = time.perf_counter()
for row in sample_array:
    _ = model.predict([row])
end_time = time.perf_counter()
avg_latency = (end_time - start_time) / len(sample_array)
```

**Step 3: Commit**
```bash
git add model/src/hyperparameter_tuning.py model/src/hyperparameter_tuning_fixed.py
git commit -m "refactor: optimize tuner latency benchmark to use NumPy representation and perf_counter"
```

---

### Task 3: Add Integration Tests for Artifact Validation and SLA

**Files:**
- Create/Modify: `tests/integration/test_artifact_sla.py`

**Step 1: Implement Integration Test**
Add a test suite that:
1. Loads the actual production model artifact (`models/optimized_lightgbm.pkl` or booster equivalent) and feature schema list (`models/feature_list.json`).
2. Loads the processed test set (`data/processed/test_enhanced.csv`).
3. Verifies that the columns of the test set align perfectly with the feature list.
4. Performs 100 predictions in sequence, measuring latency with `time.perf_counter()`, and asserts that the 95th percentile latency is strictly `< 10ms`.

**Step 2: Execute Test Suite**
Verify that the integration tests run and pass successfully:
```bash
pytest tests/integration/test_artifact_sla.py
```

**Step 3: Commit**
```bash
git add tests/integration/test_artifact_sla.py
git commit -m "test: add integration test for model artifact validation and p95 latency SLA"
```

---

### Task 4: Archive Redundant Debug Scripts and Clean Up Duplicate Data

**Files:**
- Delete: `data/raw/creditcardfraud/creditcard.csv` (and directory `data/raw/creditcardfraud/`)
- Delete/Archive: Duplicate E2E benchmark files in `debug_scripts/` (keeping only `end_to_end_test_optimized.py`)

**Step 1: Clean Up Duplicate Data**
Delete the redundant directory `data/raw/creditcardfraud/` and its nested CSV to prevent local storage waste and raw data drift. The single source of truth for raw data should be `data/raw/creditcard.csv`.

**Step 2: Remove Stale Debug Scripts**
Examine `debug_scripts/` and delete or move to an `archive/` subfolder any redundant end-to-end benchmark scripts that are no longer referenced or maintained, leaving `end_to_end_test_optimized.py` as the clean entry point.

**Step 3: Commit**
```bash
git add -A
git commit -m "chore: remove duplicate raw data folder and archive redundant E2E scripts"
```

---

### Task 5: Add CI Test Coverage Gate and Schema Versioning

**Files:**
- Modify: [`.github/workflows/ci.yml`](file:///d:/Projects/ai-ml/creditcard-fraud/.github/workflows/ci.yml)
- Modify: [`utils/validate_results.py`](file:///d:/Projects/ai-ml/creditcard-fraud/utils/validate_results.py)

**Step 1: Inject pytest-cov in CI**
Update the test job in `ci.yml` to install `pytest-cov` and execute:
```bash
pytest --cov=app --cov-fail-under=80
```
This forces all future code changes in the serving/application layer to maintain at least 80% test coverage.

**Step 2: Update validate_results.py to Support Report Schema Updates**
Update `utils/validate_results.py` to:
- Safely read and output `"schema_version"`, `"generated_at"`, and `"source_script"` fields if present in the JSON reports.
- Ensure key paths to metrics (F1, precision, recall) are robust to report format changes (e.g. support both flat keys and nested metadata blocks).

**Step 3: Commit**
```bash
git add .github/workflows/ci.yml utils/validate_results.py
git commit -m "ci: enforce 80% minimum coverage gate and update results validation script"
```

---

### Task 6: Add DATA_LICENSE File for CC BY-NC-SA 4.0 Compliance

**Files:**
- Create: `DATA_LICENSE` (at repository root)

**Step 1: Create DATA_LICENSE**
Write a file named `DATA_LICENSE` outlining that the credit card fraud dataset tracked in `data/` is licensed under CC BY-NC-SA 4.0 (Non-Commercial, Attribution, ShareAlike). Clearly state that any commercial redistribution of this data is strictly prohibited, aligning with proper licensing hygiene.

**Step 2: Commit**
```bash
git add DATA_LICENSE
git commit -m "docs: add CC BY-NC-SA 4.0 data license file"
```
