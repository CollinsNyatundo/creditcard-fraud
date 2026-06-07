"""
Unit tests for data/src/feature_engineering.py

Uses a small synthetic CSV (500 rows, ~1% fraud) so no real data is required.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.src.feature_engineering import (
    engineer_features,
    scale_features,
    create_temporal_splits,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    """500-row synthetic credit card fraud dataset."""
    rng = np.random.default_rng(42)
    n = 500
    # Simulate Time in seconds (0 to ~172800 = 2 days)
    time_vals = np.sort(rng.uniform(0, 172_800, n))
    # PCA components V1-V28
    v_cols = {f'V{i}': rng.standard_normal(n) for i in range(1, 29)}
    amount = rng.exponential(scale=50.0, size=n)
    # ~1% fraud rate
    fraud_mask = rng.uniform(size=n) < 0.01
    klass = fraud_mask.astype(int)

    df = pd.DataFrame({'Time': time_vals, **v_cols, 'Amount': amount, 'Class': klass})
    return df


# ── Tests: engineer_features ──────────────────────────────────────────────────

def test_engineer_features_adds_time_columns(synthetic_df):
    """engineer_features must add cyclical time and amount log columns."""
    result = engineer_features(synthetic_df)
    for col in ['Time_Hours', 'Time_Hour', 'Time_Hour_Sin', 'Time_Hour_Cos',
                'Amount_Log', 'Amount_Normalized']:
        assert col in result.columns, f"Missing expected column: {col}"


def test_engineer_features_no_nan(synthetic_df):
    """Engineered features must not introduce NaN values."""
    result = engineer_features(synthetic_df)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert not result[numeric_cols].isnull().any().any(), \
        "engineer_features produced NaN values in numeric columns"


def test_engineer_features_preserves_class(synthetic_df):
    """Class column must be preserved unchanged."""
    result = engineer_features(synthetic_df)
    assert 'Class' in result.columns
    assert (result['Class'].values == synthetic_df['Class'].values).all()


# ── Tests: scale_features ─────────────────────────────────────────────────────

def test_scale_features_returns_tuple(synthetic_df):
    """scale_features must return (DataFrame, scaler, list_of_features)."""
    engineered = engineer_features(synthetic_df)
    df_scaled, scaler, features = scale_features(engineered)
    assert isinstance(df_scaled, pd.DataFrame)
    assert len(features) > 0


def test_scale_features_no_nan(synthetic_df):
    """Scaling must not introduce NaN values."""
    engineered = engineer_features(synthetic_df)
    df_scaled, _, _ = scale_features(engineered)
    numeric = df_scaled.select_dtypes(include=[np.number]).columns
    assert not df_scaled[numeric].isnull().any().any()


# ── Tests: create_temporal_splits ─────────────────────────────────────────────

@pytest.fixture
def scaled_df(synthetic_df):
    engineered = engineer_features(synthetic_df)
    df_scaled, _, _ = scale_features(engineered)
    return df_scaled


def test_temporal_split_sizes(scaled_df):
    """Train/val/test sizes must approximately match 70/15/15 ratios."""
    n = len(scaled_df)
    train, val, test = create_temporal_splits(scaled_df, train_size=0.70, val_size=0.15)
    # Allow ±1 row slack due to integer rounding
    assert abs(len(train) - int(n * 0.70)) <= 1
    assert abs(len(val) - int(n * 0.15)) <= 1


def test_temporal_split_no_overlap(scaled_df):
    """The three splits must share no row indices."""
    train, val, test = create_temporal_splits(scaled_df)
    train_idx = set(train.index)
    val_idx   = set(val.index)
    test_idx  = set(test.index)
    assert train_idx.isdisjoint(val_idx), "Train and val share rows"
    assert train_idx.isdisjoint(test_idx), "Train and test share rows"
    assert val_idx.isdisjoint(test_idx), "Val and test share rows"


def test_temporal_split_chronological_order(scaled_df):
    """Temporal integrity: max(train.Time) <= min(val.Time) <= min(test.Time)."""
    train, val, test = create_temporal_splits(scaled_df)
    assert train['Time'].max() <= val['Time'].min(), (
        "Temporal leakage: train contains transactions after val starts"
    )
    assert val['Time'].max() <= test['Time'].min(), (
        "Temporal leakage: val contains transactions after test starts"
    )


def test_temporal_split_covers_all_rows(scaled_df):
    """All rows must be covered exactly once across the three splits."""
    train, val, test = create_temporal_splits(scaled_df)
    total = len(train) + len(val) + len(test)
    assert total == len(scaled_df), (
        f"Row count mismatch: expected {len(scaled_df)}, got {total}"
    )
