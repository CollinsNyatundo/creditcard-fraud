# tests/unit/test_feature_coverage_check.py
import logging
from model.src.feature_coverage_check import check_feature_explanations_coverage


def test_returns_empty_for_fully_covered_features():
    features = ["V1", "V17", "Amount", "hour_sin", "hour_cos", "rolling_mean_amount"]
    missing = check_feature_explanations_coverage(features)
    assert missing == []


def test_returns_missing_features():
    features = ["V1", "new_engineered_feature_xyz"]
    missing = check_feature_explanations_coverage(features)
    assert "new_engineered_feature_xyz" in missing
    assert "V1" not in missing


def test_logs_warning_for_missing_feature(caplog):
    with caplog.at_level(logging.WARNING, logger="model.src.feature_coverage_check"):
        check_feature_explanations_coverage(["V1", "undocumented_feature"])
    assert "undocumented_feature" in caplog.text
    assert "has no human-readable explanation" in caplog.text
