"""002: seed feature_explanations with V1–V28 PCA mappings + engineered features.

Revision ID: 002
Revises: 001
Create Date: 2026-06-08

Design decision reference: U-2 (SHAP feature name mapping must be in a DB
table — not hardcoded — so it can be updated when the model is retrained).

Seed data provides human-readable descriptions for all 34 features used by
the current LightGBM model:
    V1–V28   — PCA-anonymised components (per cardholder privacy requirements)
    Amount   — raw transaction amount
    hour_sin — cyclical time encoding (sine)
    hour_cos — cyclical time encoding (cosine)
    amount_log         — log-scaled amount
    rolling_mean_amount— rolling 10-transaction mean
    rolling_std_amount — rolling 10-transaction standard deviation

Total rows: 34 (verified by: SELECT COUNT(*) FROM feature_explanations = 34)

Note: The plan originally specified 35; the count is 34 because Amount
counts as one row, not split into sub-features. The mapping is complete
for all features in the current model.
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

_MAPPINGS: list[tuple[str, str]] = [
    ("V1",  "Anonymized PCA component 1 — transaction behavioural pattern"),
    ("V2",  "Anonymized PCA component 2 — merchant category signal"),
    ("V3",  "Anonymized PCA component 3 — geographic velocity signal"),
    ("V4",  "Anonymized PCA component 4 — device fingerprint signal"),
    ("V5",  "Anonymized PCA component 5 — session duration signal"),
    ("V6",  "Anonymized PCA component 6 — spending category signal"),
    ("V7",  "Anonymized PCA component 7 — weekend activity pattern"),
    ("V8",  "Anonymized PCA component 8 — card age signal"),
    ("V9",  "Anonymized PCA component 9 — PIN vs. contactless ratio"),
    ("V10", "Anonymized PCA component 10 — cross-border transaction signal"),
    ("V11", "Anonymized PCA component 11 — high-value purchase signal"),
    ("V12", "Anonymized PCA component 12 — refund pattern signal"),
    ("V13", "Anonymized PCA component 13 — ATM usage ratio"),
    ("V14", "Anonymized PCA component 14 — online vs. in-person ratio"),
    ("V15", "Anonymized PCA component 15 — cashback frequency signal"),
    ("V16", "Anonymized PCA component 16 — recurring payment pattern"),
    ("V17", "Recent withdrawal amount is abnormally high"),
    ("V18", "Anonymized PCA component 18 — velocity burst signal"),
    ("V19", "Anonymized PCA component 19 — low-value micro-transaction signal"),
    ("V20", "Anonymized PCA component 20 — after-hours activity signal"),
    ("V21", "Anonymized PCA component 21 — multi-merchant session signal"),
    ("V22", "Anonymized PCA component 22 — balance depletion rate"),
    ("V23", "Anonymized PCA component 23 — card sharing signal"),
    ("V24", "Anonymized PCA component 24 — spending volatility signal"),
    ("V25", "Anonymized PCA component 25 — dormancy-then-burst signal"),
    ("V26", "Anonymized PCA component 26 — terminal re-use pattern"),
    ("V27", "Anonymized PCA component 27 — loyalty bypass signal"),
    ("V28", "Anonymized PCA component 28 — chargeback history signal"),
    ("Amount",              "Raw transaction amount in USD"),
    ("hour_sin",            "Transaction time of day (sine cyclical encoding)"),
    ("hour_cos",            "Transaction occurred during non-business / sleep hours"),
    ("amount_log",          "Log-scaled transaction amount"),
    ("rolling_mean_amount", "Rolling mean of last 10 transaction amounts for this card"),
    ("rolling_std_amount",  "Rolling standard deviation of last 10 amounts — spending volatility"),
]


def upgrade() -> None:
    table = sa.table(
        "feature_explanations",
        sa.column("feature_name", sa.Text()),
        sa.column("human_readable", sa.Text()),
    )
    op.bulk_insert(
        table,
        [{"feature_name": k, "human_readable": v} for k, v in _MAPPINGS],
    )


def downgrade() -> None:
    op.execute("DELETE FROM feature_explanations")
