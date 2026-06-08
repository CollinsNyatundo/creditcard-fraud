"""Post-training guard: warn if any model feature lacks a human-readable explanation.

Resolution: U-2. Called by final_model_evaluation.py or train pipelines.
"""
import logging

logger = logging.getLogger(__name__)

# Canonical set of features expected to have explanations (from seed migration 002)
SEEDED_FEATURES = {
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "hour_sin",
    "hour_cos",
    "amount_log",
    "rolling_mean_amount",
    "rolling_std_amount",
}


def check_feature_explanations_coverage(model_feature_names: list[str]) -> list[str]:
    """Return list of features that have NO entry in feature_explanations.

    Logs a WARNING for each missing feature.
    As agreed in design lock, this only warns and does not block model serialization.
    """
    missing = [f for f in model_feature_names if f not in SEEDED_FEATURES]
    for feature in missing:
        logger.warning(
            "Feature '%s' has no human-readable explanation in feature_explanations table. "
            "Add an entry via Alembic migration before deploying.",
            feature,
        )
    if not missing:
        logger.info("Feature explanation coverage check: 100%% complete. All features documented.")
    return missing
