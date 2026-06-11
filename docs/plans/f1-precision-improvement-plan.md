# F1 and Precision Improvement Plan

## Current State
- Test F1: 0.8041
- Test Precision: 0.8667
- Test Recall: 0.75
- Gap to F1 target (0.85): ~0.046
- Model: LightGBM with focal loss and Optuna tuning already applied
- Latency: well under 10ms

## Improvement Strategy
1. Calibrate probabilities to improve Precision-Recall tradeoff
2. Add F1-Precision Pareto-aware Optuna objective
3. Constrain threshold search for better Precision-F1 balance
4. Add feature selection/robustness to reduce variance
5. Evaluate and document metrics

## Tasks

### Task 1: Add probability calibration for post-training refinement
Create `model/src/calibrate_probabilities.py` that:
- Loads the current model and validation/test data
- Fits `CalibratedClassifierCV` with `method="isotonic"` on validation set
- Saves calibrated model wrapper and updated probabilities
- Outputs before/after Precision, Recall, F1 on test set

### Task 2: Create Pareto-aware Optuna objective
Update `model/src/hyperparameter_tuning.py` to:
- Add optional `'f1_precision_pareto'` objective that optimizes a weighted combination
- Use `min(1.0, precision / 0.90)` as precision penalty factor
- Combine as `f1 * precision_penalty` to reward both metrics (higher is better)
- Keep existing focal loss objective as fallback
- Use `maximize` optuna direction for pareto mode
### Task 3: Add constrained threshold search
Create `model/src/optimize_threshold_pareto.py` that:
- Searches thresholds from 0.1 to 0.9 with 0.01 step
- Filters thresholds where Precision >= 0.85
- From remaining, selects threshold with highest F1
- Falls back to highest Precision if F1 constraint cannot be met
- Saves optimized threshold and metrics to JSON

### Task 4: Add feature importance and selection utility
Create `model/src/feature_selection.py` that:
- Extracts feature importances from trained model
- Identifies top-N features by importance
- Tests model performance with reduced feature set
- Reports Precision, Recall, F1 comparison

### Task 5: Re-run evaluation with improvements
Update `model/src/final_model_evaluation.py` to:
- Load calibrated probabilities if available
- Apply Pareto-optimized threshold if available
- Compare baseline vs optimized metrics side-by-side
- Generate improvement report

## Implementation Order
Task 1 → Task 2 → Task 3 → Task 4 → Task 5

Each task should:
1. Write minimal code to implement the feature
2. Run tests/verification
3. Save results
4. Move to next task
