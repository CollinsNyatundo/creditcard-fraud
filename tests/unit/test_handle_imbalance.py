"""
Unit tests for data/src/handle_imbalance.py

Uses a small synthetic training DataFrame so no disk I/O to real data is needed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.src.handle_imbalance import handle_class_imbalance, analyze_class_distribution


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def imbalanced_train_df():
    """Small synthetic training DataFrame with ~0.5% fraud rate."""
    rng = np.random.default_rng(0)
    n = 2000
    v_cols = {f'V{i}': rng.standard_normal(n) for i in range(1, 29)}
    amount = rng.exponential(scale=50.0, size=n)
    # 10 fraud cases (~0.5% rate)
    klass = np.zeros(n, dtype=int)
    klass[:10] = 1
    rng.shuffle(klass)
    df = pd.DataFrame({**v_cols, 'Amount': amount, 'Class': klass})
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_handle_imbalance_output_fraud_ratio(imbalanced_train_df):
    """After balancing, fraud-to-non-fraud ratio must be approximately 1:5."""
    balanced, _ = handle_class_imbalance(imbalanced_train_df, random_state=42)
    counts = balanced['Class'].value_counts()
    fraud     = counts.get(1, 0)
    non_fraud = counts.get(0, 0)
    assert fraud > 0, "Balanced dataset contains no fraud samples"
    ratio = non_fraud / fraud
    # Target 1:5 (non_fraud:fraud = 5), allow ±30% tolerance
    assert 3.5 <= ratio <= 6.5, (
        f"Expected fraud ratio ~1:5, got 1:{ratio:.2f}"
    )


def test_handle_imbalance_increases_fraud_count(imbalanced_train_df):
    """SMOTE must increase the number of fraud samples above the original."""
    original_fraud = imbalanced_train_df['Class'].sum()
    balanced, _ = handle_class_imbalance(imbalanced_train_df, random_state=42)
    balanced_fraud = balanced['Class'].sum()
    assert balanced_fraud > original_fraud, (
        "SMOTE did not increase fraud sample count"
    )


def test_handle_imbalance_returns_dataframe(imbalanced_train_df):
    """handle_class_imbalance must return a DataFrame with 'Class' column."""
    balanced, preprocessor = handle_class_imbalance(imbalanced_train_df, random_state=42)
    assert isinstance(balanced, pd.DataFrame)
    assert 'Class' in balanced.columns


def test_handle_imbalance_no_nan(imbalanced_train_df):
    """Balanced DataFrame must not contain NaN values."""
    balanced, _ = handle_class_imbalance(imbalanced_train_df, random_state=42)
    assert not balanced.isnull().any().any(), "Balanced dataset contains NaN values"


def test_analyze_class_distribution_runs(imbalanced_train_df, capsys):
    """analyze_class_distribution must run without error and print output."""
    analyze_class_distribution(imbalanced_train_df, "TEST SET")
    captured = capsys.readouterr()
    assert "Class" in captured.out or "class" in captured.out.lower()
