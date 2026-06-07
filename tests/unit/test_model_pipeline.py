"""
Unit tests for model pipeline — serialization round-trip and basic predictions.

Uses a tiny synthetic LightGBM model to avoid any disk I/O to real trained models.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_tiny_dataset(n: int = 200, n_features: int = 10, seed: int = 42):
    """Create a tiny labeled dataset for quick model training."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    # Binary target with ~20% positive rate
    y = (rng.uniform(size=n) < 0.20).astype(int)
    return X, y


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_lightgbm_imports():
    """LightGBM must be importable at module level."""
    import lightgbm as lgb  # noqa: F401
    assert hasattr(lgb, 'LGBMClassifier')


def test_model_round_trip_predictions():
    """Serialize model to disk, reload it, and verify predictions are identical."""
    import joblib
    import lightgbm as lgb

    X, y = make_tiny_dataset(n=200, n_features=10)

    model = lgb.LGBMClassifier(
        n_estimators=10,
        num_leaves=4,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    preds_before = model.predict_proba(X)[:, 1]

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name

    try:
        joblib.dump(model, tmp_path)
        loaded_model = joblib.load(tmp_path)
        preds_after = loaded_model.predict_proba(X)[:, 1]
    finally:
        os.unlink(tmp_path)

    np.testing.assert_array_equal(
        preds_before, preds_after,
        err_msg="Predictions changed after model serialization round-trip"
    )


def test_model_predict_shape():
    """Model predict output shape must match number of input rows."""
    import lightgbm as lgb

    X, y = make_tiny_dataset(n=100, n_features=5)
    model = lgb.LGBMClassifier(n_estimators=5, num_leaves=4, random_state=0, verbose=-1)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (100, 2), f"Expected (100, 2), got {proba.shape}"


def test_model_feature_names_preserved():
    """Feature names stored in the model must match training column names."""
    import lightgbm as lgb

    X, y = make_tiny_dataset(n=100, n_features=8)
    model = lgb.LGBMClassifier(n_estimators=5, num_leaves=4, random_state=0, verbose=-1)
    model.fit(X, y)

    assert list(model.feature_name_) == list(X.columns), \
        "Model feature names do not match training DataFrame columns"


def test_latency_single_row():
    """Single-row inference must complete in under 100 ms (generous local bound)."""
    import time
    import lightgbm as lgb

    X, y = make_tiny_dataset(n=300, n_features=10)
    model = lgb.LGBMClassifier(n_estimators=20, num_leaves=8, random_state=0, verbose=-1)
    model.fit(X, y)

    single_row = X.iloc[0:1]
    start = time.perf_counter()
    _ = model.predict_proba(single_row)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 100, (
        f"Single-row inference took {elapsed_ms:.1f} ms — expected < 100 ms"
    )
