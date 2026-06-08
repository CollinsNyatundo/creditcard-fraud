"""001: canonical schema — predictions, shap_explanations, system_config, api_keys, feature_explanations.

Revision ID: 001
Revises: (none — initial migration)
Create Date: 2026-06-08

This migration creates the five core tables required by the Phase 1 architecture.
All schema decisions are documented in docs/design_decisions.md (C-2).

Tables:
    predictions         — one row per inference call
    shap_explanations   — async SHAP values for flagged transactions
    system_config       — runtime key-value config (replaces hardcoded thresholds)
    api_keys            — hashed API key registry (C-1)
    feature_explanations— SHAP-to-English translation table (U-2)

Indexes:
    idx_predictions_card_id   — fast lookup by card
    idx_predictions_created_at— fast range scans for WebSocket replay (U-4)
    idx_shap_prediction_id    — fast join from predictions → shap_explanations
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgcrypto for gen_random_uuid()
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # --- predictions ---
    op.create_table(
        "predictions",
        sa.Column(
            "id",
            UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("card_id", sa.Text(), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("is_flagged", sa.Boolean(), nullable=False),
        sa.Column("threshold_used", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("idx_predictions_card_id", "predictions", ["card_id"])
    op.create_index(
        "idx_predictions_created_at",
        "predictions",
        [sa.text("created_at DESC")],
    )

    # --- shap_explanations ---
    op.create_table(
        "shap_explanations",
        sa.Column(
            "id",
            UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "prediction_id",
            UUID(),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column("shap_value", sa.Float(), nullable=False),
        sa.Column("human_readable", sa.Text()),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_shap_prediction_id", "shap_explanations", ["prediction_id"]
    )

    # --- system_config ---
    op.create_table(
        "system_config",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.bulk_insert(
        sa.table(
            "system_config",
            sa.column("key", sa.Text()),
            sa.column("value", sa.Text()),
        ),
        [
            {"key": "shap_trigger_threshold", "value": "0.50"},
            {"key": "alert_threshold", "value": "0.90"},
            {"key": "alert_enabled", "value": "false"},
            {"key": "alert_webhook_url", "value": ""},
        ],
    )

    # --- api_keys ---
    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("key_hash", sa.Text(), unique=True, nullable=False),
        sa.Column("label", sa.Text()),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )

    # --- feature_explanations ---
    op.create_table(
        "feature_explanations",
        sa.Column("feature_name", sa.Text(), primary_key=True),
        sa.Column("human_readable", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("feature_explanations")
    op.drop_table("api_keys")
    op.drop_table("system_config")
    op.drop_index("idx_shap_prediction_id", table_name="shap_explanations")
    op.drop_table("shap_explanations")
    op.drop_index("idx_predictions_created_at", table_name="predictions")
    op.drop_index("idx_predictions_card_id", table_name="predictions")
    op.drop_table("predictions")
