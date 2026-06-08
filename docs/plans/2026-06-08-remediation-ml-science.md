# ML Science & Optimization Remediation — Implementation Plan

> **For Agent:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Resolve the critical ML science issues identified in the production audit report: threshold optimization leakage, shuffled cross-validation temporal contradictions, and insufficient Optuna hyperparameter trials.

**Architecture:** 
1. Replace StratifiedKFold with TimeSeriesSplit in `hyperparameter_tuning_fixed.py` to preserve temporal integrity of validation.
2. Freeze decision thresholds on the validation folds and perform a single-pass evaluation on held-out test data to eliminate threshold optimization leakage.
3. Scale Optuna trials from 5 to 50+ using LightGBM's pruning callbacks to search the 8-parameter space thoroughly.
4. Correct the runtime Booster crash in `final_model_evaluation.py`.

**Tech Stack:** Python 3.10+ · LightGBM · scikit-learn · Optuna · MLflow

**Design Decisions / Audit Reference:** 
- Production Audit Report: P0-1, P0-2, P0-4, P1-1, P1-3
- Claims Validity Matrix: Item 1, Item 2, Item 4, Item 8

---

## Mandatory Gate

> ⚠️ **Do not begin this remediation plan until the implementation plan itself is approved by the user.**
> ⚠️ **All tasks must end with a single semantic git commit. Do not squash multiple tasks into one commit.**

---

### Task 1: Replace StratifiedKFold with TimeSeriesSplit in Tuning

**Files:**
- Modify: [`model/src/hyperparameter_tuning_fixed.py`](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/hyperparameter_tuning_fixed.py)

**Step 1: Replace Fold Splitter**
Modify the objective function to use `TimeSeriesSplit(n_splits=3)` (or 5) instead of shuffled `StratifiedKFold`. This guarantees that training data always chronologically precedes validation data.

```python
from sklearn.model_selection import TimeSeriesSplit

# Replace:
# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# With:
tscv = TimeSeriesSplit(n_splits=3)
```

Ensure the features and target are split sequentially using `tscv.split(X_train_val)`. Since `TimeSeriesSplit` does not take `y` for stratification, use `tscv.split(X_train_val)`.

**Step 2: Verify Split Order**
Add assertions inside the fold loop confirming that the maximum index in the training fold is strictly less than the minimum index in the validation fold:
```python
assert train_idx.max() < val_idx.min(), "Temporal order violated in TimeSeriesSplit folds"
```

**Step 3: Commit**
```bash
git add model/src/hyperparameter_tuning_fixed.py
git commit -m "feat: replace StratifiedKFold with TimeSeriesSplit in hyperparameter tuning"
```

---

### Task 2: Freeze Decision Threshold on Validation Set (Fix Test Leakage)

**Files:**
- Modify: [`model/src/hyperparameter_tuning_fixed.py`](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/hyperparameter_tuning_fixed.py)

**Step 1: Shift Threshold Optimization to Validation Folds**
In the objective function or final model evaluation step, extract validation predictions and run the threshold optimizer *only* on the validation labels and predictions:
```python
# Optimize threshold on validation data
val_preds = model.predict(X_val) # or predict_proba
best_threshold, _ = optimize_threshold(y_val, val_preds)
```

**Step 2: Evaluate on Test Set Using Frozen Threshold**
When generating test metrics (like F1, Precision, Recall) for the final model or trials, evaluate the test predictions *strictly* against this frozen `best_threshold`. Do not optimize on test labels `y_test`:
```python
# Single evaluation on test set using validation-optimized threshold
test_preds = (test_proba >= best_threshold).astype(int)
# Compute F1, Precision, Recall on test_preds vs y_test
```

**Step 3: Commit**
```bash
git add model/src/hyperparameter_tuning_fixed.py
git commit -m "fix: optimize decision threshold on validation set instead of test set"
```

---

### Task 3: Increase Optuna trials to 50+ & Add LightGBM Pruning Callback

**Files:**
- Modify: [`model/src/hyperparameter_tuning_fixed.py`](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/hyperparameter_tuning_fixed.py)

**Step 1: Increase Trial Count**
Change the `n_trials` parameter of the study execution from 5 to 50 (or 100) to allow proper exploration of the 8-parameter LightGBM search space.

**Step 2: Integrate Optuna Pruning Callback**
Add the `LightGBMPruningCallback` to early-stop poorly performing trials to keep execution times reasonable:
```python
import optuna
from optuna.integration import LightGBMPruningCallback

# In the training callback list:
pruning_callback = LightGBMPruningCallback(trial, "binary_logloss")
# Pass to model fit method: callbacks=[pruning_callback]
```

**Step 3: Commit**
```bash
git add model/src/hyperparameter_tuning_fixed.py
git commit -m "feat: scale Optuna trials to 50 and add LightGBM pruning callback"
```

---

### Task 4: Fix `final_model_evaluation.py` Runtime booster Crash

**Files:**
- Modify: [`model/src/final_model_evaluation.py`](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/final_model_evaluation.py)

**Step 1: Replace `predict_proba` with `predict`**
In `model/src/final_model_evaluation.py` (specifically around line 66), change the prediction call on the LightGBM `Booster` model. LightGBM Booster models do not have a `predict_proba` method like the scikit-learn wrapper does; they output probabilities directly via `predict()`.

```python
# Replace:
# y_pred_proba = model.predict_proba(X_test)
# With:
y_pred_proba = model.predict(X_test)
```

**Step 2: Verify Script Run**
Execute the model evaluation script locally to confirm it runs successfully without throwing `AttributeError`:
```bash
python model/src/final_model_evaluation.py
```

**Step 3: Commit**
```bash
git add model/src/final_model_evaluation.py
git commit -m "fix: replace predict_proba with predict for LightGBM Booster evaluation"
```

---

### Task 5: Consolidate Hyperparameter Tuning Script and Update Makefile

**Files:**
- Modify: [`Makefile`](file:///d:/Projects/ai-ml/creditcard-fraud/Makefile)
- Modify: [`model/src/hyperparameter_tuning_fixed.py`](file:///d:/Projects/ai-ml/creditcard-fraud/model/src/hyperparameter_tuning_fixed.py) (Renamed to `model/src/hyperparameter_tuning.py`)
- Delete: `model/src/hyperparameter_tuning.py` (Old file)

**Step 1: Overwrite/Rename Tuning Script**
Delete the old `model/src/hyperparameter_tuning.py` script. Rename `model/src/hyperparameter_tuning_fixed.py` to `model/src/hyperparameter_tuning.py` so that it replaces the old script.

**Step 2: Update Makefile**
Update the model tuning target in `Makefile` to run the consolidated `hyperparameter_tuning.py` script:
```makefile
# Replace:
# $(PYTHON) model/src/hyperparameter_tuning_fixed.py
# With:
$(PYTHON) model/src/hyperparameter_tuning.py
```

**Step 3: Verify**
Run the consolidated script using the Makefile to verify:
```bash
make tune-hyperparameters
```

**Step 4: Commit**
```bash
git rm model/src/hyperparameter_tuning.py
git mv model/src/hyperparameter_tuning_fixed.py model/src/hyperparameter_tuning.py
git add Makefile model/src/hyperparameter_tuning.py
git commit -m "refactor: consolidate hyperparameter tuning scripts and update Makefile"
```
