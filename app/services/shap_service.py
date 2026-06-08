import asyncio
import numpy as np
import shap
import sqlalchemy as sa
from app.db.engine import AsyncSessionLocal


async def compute_and_store_shap(
    prediction_id: str,
    model,
    features,
    feature_names: list[str],
) -> None:
    """Calculate local SHAP values and write to shap_explanations table.
    Called only for transactions above the shap_trigger_threshold.
    """
    explainer = await asyncio.to_thread(shap.TreeExplainer, model)
    shap_vals = await asyncio.to_thread(explainer.shap_values, features)

    # Support shap.Explanation objects (common in modern shap)
    if hasattr(shap_vals, "values"):
        shap_vals = shap_vals.values

    # Normalize list / array shapes for binary classification
    if isinstance(shap_vals, list):
        if len(shap_vals) > 1:
            # Binary classification list [neg_class, pos_class]
            shap_values = shap_vals[1]
        else:
            shap_values = shap_vals[0]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        if shap_vals.shape[0] == 2:
            shap_values = shap_vals[1]
        elif shap_vals.shape[2] == 2:
            shap_values = shap_vals[:, :, 1]
        else:
            shap_values = shap_vals
    else:
        shap_values = shap_vals

    # Extract first sample's 1D vector if shape is 2D
    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # Fetch human-readable name map from database
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa.text("SELECT feature_name, human_readable FROM feature_explanations")
        )
        name_map = {row[0]: row[1] for row in result.fetchall() if row[0] is not None}

    rows = [
        {
            "prediction_id": prediction_id,
            "feature_name": name,
            "shap_value": float(val),
            "human_readable": name_map.get(name),
        }
        for name, val in zip(feature_names, shap_values)
    ]

    # Batch insert features explanation into database
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                sa.text("""
                    INSERT INTO shap_explanations
                      (prediction_id, feature_name, shap_value, human_readable)
                    VALUES (:prediction_id, :feature_name, :shap_value, :human_readable)
                """),
                rows,
            )
