import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import time
import json
import warnings
from scipy import optimize, special

warnings.filterwarnings('ignore')

class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y_true):
        if self.alpha is None:
            return 1.0
        y_true = np.asarray(y_true)
        return np.where(y_true == 1, self.alpha, 1.0 - self.alpha)

    def pt(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return np.where(y_true == 1, y_pred, 1.0 - y_pred)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1.0 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y = 2 * y_true - 1
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1.0 - pt) ** g * (g * pt * np.log(pt) + pt - 1.0)

    def hess(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y = 2 * y_true - 1
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        u = at * y * (1.0 - pt) ** g
        du = -at * y * g * (1.0 - pt) ** (g - 1.0)
        v = g * pt * np.log(pt) + pt - 1.0
        dv = g * np.log(pt) + g + 1.0
        return (du * v + u * dv) * y * (pt * (1.0 - pt))

    def init_score(self, y_true):
        y_true = np.asarray(y_true)
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(1e-15, 1.0 - 1e-15),
            method='bounded'
        )
        p = res.x
        return np.log(p / (1.0 - p))

    def lgb_obj(self, preds, dtrain):
        y_true = dtrain.get_label()
        p = special.expit(preds)
        return self.grad(y_true, p), self.hess(y_true, p)

    def lgb_eval(self, preds, dtrain):
        y_true = dtrain.get_label()
        p = special.expit(preds)
        loss = self(y_true, p)
        return 'focal_loss', np.mean(loss), False

class OptunaPruningCallback:
    def __init__(self, trial, metric_name):
        self.trial = trial
        self.metric_name = metric_name

    def __call__(self, env):
        for _, eval_name, val, is_higher_better in env.evaluation_result_list:
            if eval_name == self.metric_name:
                step = env.iteration
                study_direction = self.trial.study.direction
                if (study_direction == optuna.study.StudyDirection.MAXIMIZE and not is_higher_better) or \
                   (study_direction == optuna.study.StudyDirection.MINIMIZE and is_higher_better):
                    report_val = -val
                else:
                    report_val = val
                
                self.trial.report(report_val, step)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

class LatencyConstrainedObjective:
    def __init__(self, X_train, y_train, X_val, y_val, feature_cols, latency_threshold=0.008):
        self.X_train = X_train[feature_cols]
        self.y_train = y_train
        self.X_val = X_val[feature_cols]
        self.y_val = y_val
        self.feature_cols = feature_cols
        self.latency_threshold = latency_threshold  # 8ms in seconds

    def __call__(self, trial):
        # Define hyperparameter search space
        alpha = trial.suggest_float('alpha', 0.1, 0.9)
        gamma = trial.suggest_float('gamma', 0.5, 5.0)

        focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'objective': focal_loss.lgb_obj,  # Pass custom objective callable directly here
            'metric': 'None',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        }

        # Cross-validation with TimeSeriesSplit folds
        tscv = TimeSeriesSplit(n_splits=3)
        f1_scores = []
        latencies = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            assert train_idx.max() < val_idx.min(), "Temporal order violated in TimeSeriesSplit folds"
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            init_score_val = focal_loss.init_score(y_fold_train)

            # Train model
            train_data = lgb.Dataset(
                X_fold_train,
                label=y_fold_train,
                init_score=np.full_like(y_fold_train, init_score_val, dtype=float)
            )
            valid_data = lgb.Dataset(
                X_fold_val,
                label=y_fold_val,
                init_score=np.full_like(y_fold_val, init_score_val, dtype=float),
                reference=train_data
            )

            pruning_callback = OptunaPruningCallback(trial, 'focal_loss')
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                feval=focal_loss.lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), pruning_callback]
            )

            # Predictions
            raw_preds = model.predict(X_fold_val, num_iteration=model.best_iteration)
            y_pred_proba = special.expit(raw_preds + init_score_val)

            # Optimize threshold on validation fold
            _, fold_f1 = optimize_threshold(y_fold_val, y_pred_proba)
            f1_scores.append(fold_f1)

            # Measure latency on a sample of predictions
            sample_indices = np.random.choice(len(X_fold_val), min(100, len(X_fold_val)), replace=False)
            sample_X = X_fold_val.iloc[sample_indices]
            start_time = time.time()
            for _, row in sample_X.iterrows():
                pred_raw = model.predict([row], num_iteration=model.best_iteration)
                _ = special.expit(pred_raw + init_score_val)
            end_time = time.time()
            avg_latency = (end_time - start_time) / len(sample_X)
            latencies.append(avg_latency)

        avg_f1 = np.mean(f1_scores)
        avg_latency = np.mean(latencies)

        if avg_latency > self.latency_threshold:
            return float('-inf')
        return avg_f1

def load_data():
    """Load enhanced datasets"""
    print("Loading enhanced datasets...")
    train_df = pd.read_csv('./data/processed/train_enhanced.csv')
    val_df = pd.read_csv('./data/processed/val_enhanced.csv')
    test_df = pd.read_csv('./data/processed/test_enhanced.csv')

    # Get all columns except Class and Time
    feature_cols = [col for col in train_df.columns if col not in ['Class', 'Time']]

    # Make sure all datasets have the same features
    common_features = set(feature_cols)
    common_features = common_features.intersection(set(val_df.columns))
    common_features = common_features.intersection(set(test_df.columns))
    feature_cols = list(common_features)
    print(f"Using {len(feature_cols)} common features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['Class']
    X_val = val_df[feature_cols]
    y_val = val_df['Class']
    X_test = test_df[feature_cols]
    y_test = test_df['Class']
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def optimize_threshold(y_true, y_pred_proba):
    """Find optimal threshold to maximize F1 score"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, f1_scores[optimal_idx]

def evaluate_model(model, X_val, y_val, X_test, y_test, feature_cols, init_score_val):
    """Evaluate model performance and measure latency"""
    print("Evaluating final model...")
    # Optimize threshold on validation set (no leakage)
    y_val_raw = model.predict(X_val[feature_cols], num_iteration=model.best_iteration)
    y_val_proba = special.expit(y_val_raw + init_score_val)
    optimal_threshold, _ = optimize_threshold(y_val, y_val_proba)
    print(f"Optimal threshold found on validation set: {optimal_threshold:.4f}")

    # Evaluate on test set using the validation-optimized threshold
    y_test_raw = model.predict(X_test[feature_cols], num_iteration=model.best_iteration)
    y_pred_proba = special.expit(y_test_raw + init_score_val)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Measure latency
    sample_indices = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
    sample_X = X_test.iloc[sample_indices]
    start_time = time.time()
    for _, row in sample_X.iterrows():
        pred_raw = model.predict([row], num_iteration=model.best_iteration)
        _ = special.expit(pred_raw + init_score_val)
    end_time = time.time()
    avg_latency = (end_time - start_time) / len(sample_X)

    # Measure 95th percentile latency
    latency_samples = []
    for i in range(min(100, len(sample_X))):
        start = time.time()
        pred_raw = model.predict([sample_X.iloc[i]], num_iteration=model.best_iteration)
        _ = special.expit(pred_raw + init_score_val)
        end = time.time()
        latency_samples.append(end - start)
    latency_95_percentile = np.percentile(latency_samples, 95)

    results = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': auc,
        'avg_latency': avg_latency,
        'latency_95_percentile': latency_95_percentile,
        'optimal_threshold': optimal_threshold
    }
    return results

def main():
    """Main function to perform hyperparameter tuning"""
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()

    # Create objective function
    objective = LatencyConstrainedObjective(X_train, y_train, X_val, y_val, feature_cols)

    # Create study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

    # Optimize
    print("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=150, timeout=1800)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")

    # Train final model with best parameters
    print("Training final model with best parameters...")
    best_params = study.best_trial.params.copy()
    best_alpha = best_params.pop('alpha')
    best_gamma = best_params.pop('gamma')

    focal_loss_final = FocalLoss(alpha=best_alpha, gamma=best_gamma)

    best_params.update({
        'objective': focal_loss_final.lgb_obj,  # Pass custom objective callable directly here
        'metric': 'None',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42
    })

    # Combine train and validation sets for final training
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    final_init_score = focal_loss_final.init_score(y_combined)

    train_data = lgb.Dataset(
        X_combined[feature_cols],
        label=y_combined,
        init_score=np.full_like(y_combined, final_init_score, dtype=float)
    )
    valid_data = lgb.Dataset(
        X_val[feature_cols],
        label=y_val,
        init_score=np.full_like(y_val, final_init_score, dtype=float),
        reference=train_data
    )

    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        feval=focal_loss_final.lgb_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Evaluate model
    results = evaluate_model(final_model, X_val, y_val, X_test, y_test, feature_cols, final_init_score)
    print("\n=== MODEL EVALUATION RESULTS ===")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Average Latency: {results['avg_latency'] * 1000:.2f}ms")
    print(f"95th Percentile Latency: {results['latency_95_percentile'] * 1000:.2f}ms")
    print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")

    # Check if requirements are met
    if results['f1_score'] > 0.85 and results['avg_latency'] < 0.008:
        print("\n[OK] Model meets requirements: F1 > 0.85 and latency < 8ms")
    else:
        print("\n[FAIL] Model does not meet requirements")

    # Save model and results
    print("Saving model and results...")
    final_model.save_model('./models/optimized_lightgbm.txt')
    joblib.dump(final_model, './models/optimized_lightgbm.pkl')

    # Save optimal threshold and focal loss config
    with open('./models/optimal_threshold_v2.json', 'w') as f:
        json.dump({
            'threshold': results['optimal_threshold'],
            'init_score': float(final_init_score),
            'alpha': float(best_alpha),
            'gamma': float(best_gamma),
            'is_focal_loss': True
        }, f)

    # Save feature list
    with open('./models/feature_list.json', 'w') as f:
        json.dump(feature_cols, f)

    # Save optimization results
    optimization_results = {
        "schema_version": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "generated_at": pd.Timestamp.now().isoformat(),
        "source_script": "model/src/hyperparameter_tuning.py",
        "artifact_of": "creditcard-fraud-pipeline",
        'best_params': study.best_trial.params,
        'best_f1': study.best_trial.value,
        'final_metrics': results
    }
    with open('./reports/hyperparameter_optimization.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)
    print("Hyperparameter tuning completed successfully.")

if __name__ == "__main__":
    main()
