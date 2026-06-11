"""tests/unit/test_threshold_optimization.py

Unit tests for model/src/optimize_threshold_pareto.py — dual-threshold
selection and cost calculation logic.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap so tests resolve project modules without an installed package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.src.optimize_threshold_pareto import (
    calculate_cost,
    search_dual_thresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(probs: np.ndarray) -> MagicMock:
    """Return a mock model whose predict() returns the given probability array."""
    m = MagicMock()
    del m.predict_proba          # force the raw .predict() code-path
    m.predict.return_value = probs
    return m


def _make_proba_model(probs: np.ndarray) -> MagicMock:
    """Return a mock calibrated model whose predict_proba() returns class probs."""
    m = MagicMock()
    m.predict_proba.return_value = np.column_stack([1 - probs, probs])
    return m


# ---------------------------------------------------------------------------
# Test 1 — cost function calculation (R12)
# ---------------------------------------------------------------------------

class TestCalculateCost:
    """Unit tests for the financial cost function."""

    def test_pure_false_negatives(self):
        """FN cost = amount + $15 per missed fraud."""
        y_true = np.array([1, 1, 0])
        y_pred = np.array([0, 0, 0])          # 2 FNs, 0 FPs
        amounts = np.array([100.0, 200.0, 50.0])

        cost = calculate_cost(y_true, y_pred, amounts)

        expected = (100.0 + 15.0) + (200.0 + 15.0)   # = 330.0
        assert cost == pytest.approx(expected)

    def test_pure_false_positives(self):
        """FP cost = $50 flat per false alarm."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 0])          # 0 FNs, 2 FPs
        amounts = np.array([10.0, 20.0, 30.0])

        cost = calculate_cost(y_true, y_pred, amounts)

        expected = 2 * 50.0                    # = 100.0
        assert cost == pytest.approx(expected)

    def test_mixed_fn_and_fp(self):
        """Combined FN + FP cost formula is correct."""
        y_true   = np.array([1,  0,  1,  0])
        y_pred   = np.array([0,  1,  1,  0])
        amounts  = np.array([80.0, 0.0, 200.0, 0.0])

        # FN: index 0 → 80 + 15 = 95
        # FP: index 1 → 50
        cost = calculate_cost(y_true, y_pred, amounts)
        assert cost == pytest.approx(95.0 + 50.0)

    def test_perfect_predictions_zero_cost(self):
        """Zero cost when there are no FNs and no FPs."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        amounts = np.array([100.0, 50.0, 200.0, 30.0])

        assert calculate_cost(y_true, y_pred, amounts) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 2 — recall-constrained threshold found (R1 / R2)
# ---------------------------------------------------------------------------

class TestSearchDualThresholds:
    """Tests for the dual-threshold search: cost_min and recall_target."""

    def _build_dataset(self, n_samples: int = 200, fraud_rate: float = 0.1):
        """Build a synthetic labelled dataset and matching probabilities.

        Fraud samples get high probabilities; legit samples get low ones,
        so that there exists a threshold that achieves Recall >= 0.85.
        """
        rng = np.random.default_rng(0)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        y = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
        # Probabilities: fraud 0.7–1.0, legit 0.0–0.3
        probs = np.concatenate([
            rng.uniform(0.70, 1.00, n_fraud),
            rng.uniform(0.00, 0.30, n_legit),
        ])
        amounts = rng.uniform(10, 500, n_samples)
        X = pd.DataFrame({"Amount": amounts})
        return X, y, probs, amounts

    def test_recall_target_threshold_found(self):
        """recall_target must satisfy Recall >= 0.85 when the data allows it."""
        X, y, probs, amounts = self._build_dataset()
        model = _make_proba_model(probs)

        cost_min, recall_target, results, _, _ = search_dual_thresholds(
            model, X, y, amounts
        )

        # Verify recall at the reported threshold
        preds = (probs >= recall_target).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        actual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert actual_recall >= 0.85, (
            f"Expected recall_target={recall_target} to yield Recall >= 0.85, "
            f"but got {actual_recall:.4f}"
        )

    def test_recall_constraint_unachievable_fallback(self, caplog):
        """When no threshold reaches Recall >= 0.85, the function falls back to
        the best-effort threshold and emits a WARNING; no exception is raised.

        Data construction: all 10 fraud samples have probabilities capped at 0.30,
        while the 90 legit samples span 0.01-0.99.  At threshold=0.01 every sample
        is predicted fraud (Recall=1.0 — achievable!), so we must instead make
        the fraud probabilities HIGHER than legit ones but clamp the total so that
        even at threshold=0.01, fewer than 85% of frauds are caught.

        The cleanest approach: put ALL fraud samples at prob=0.0 (below any threshold
        that would be applied) so they are always missed → Recall is always 0.
        """
        import logging

        n_fraud = 10
        n_legit = 90
        y = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
        # Fraud probability = 0.0 at every threshold → always predicted negative
        # → Recall is 0 at every threshold → constraint cannot be met
        probs = np.zeros(n_fraud + n_legit)
        amounts = np.ones(n_fraud + n_legit) * 100.0
        X = pd.DataFrame({"Amount": amounts})
        model = _make_proba_model(probs)

        with caplog.at_level(logging.WARNING, logger="model.src.optimize_threshold_pareto"):
            cost_min, recall_target, results, _, _ = search_dual_thresholds(
                model, X, y, amounts
            )

        # A warning mentioning the 0.85 gate must have been emitted
        assert any("0.85" in record.message for record in caplog.records), (
            "Expected a WARNING about Recall >= 0.85 being unachievable, "
            f"but log records were: {[r.message for r in caplog.records]}"
        )
        # Function must still return valid floats without raising
        assert isinstance(cost_min, float)
        assert isinstance(recall_target, float)


    def test_json_output_schema(self):
        """save_results must write JSON with cost_min, recall_target, active,
        threshold — per the R13/R14 output schema requirement."""
        from model.src.optimize_threshold_pareto import save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            reports_dir = Path(tmpdir) / "reports"
            reports_dir.mkdir()

            # Patch the module-level MODELS_DIR and REPORTS_DIR constants
            with patch("model.src.optimize_threshold_pareto.MODELS_DIR", models_dir), \
                 patch("model.src.optimize_threshold_pareto.REPORTS_DIR", reports_dir), \
                 patch("model.src.optimize_threshold_pareto.asyncio.run"):  # skip DB sync

                save_results(
                    cost_min=0.35,
                    recall_target=0.22,
                    metrics={
                        "cost_min": 0.35,
                        "recall_target": 0.22,
                        "metrics_cost_min": {"recall": 0.80, "cost": 1234.0},
                        "metrics_recall_target": {"recall": 0.87, "cost": 2500.0},
                    },
                )

            output_path = models_dir / "optimal_threshold_v2.json"
            assert output_path.exists(), "optimal_threshold_v2.json was not written"
            with output_path.open() as fh:
                data = json.load(fh)

        required_keys = {"cost_min", "recall_target", "active", "threshold"}
        missing = required_keys - set(data.keys())
        assert not missing, f"JSON missing required keys: {missing}"
        # threshold must be the resolved value of the active key
        assert data["threshold"] == data[data["active"]]
