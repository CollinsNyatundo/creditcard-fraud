"""Idempotent Metabase provisioning script.

Usage:
    python scripts/setup_metabase.py

Env vars required:
    METABASE_URL    — e.g. http://localhost:3000
    METABASE_EMAIL  — admin email
    METABASE_PASS   — admin password
    DATABASE_URL    — PostgreSQL connection string (parsed for host/port/db)

Resolution: U-3
"""
import asyncio
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

METABASE_URL = os.environ.get("METABASE_URL", "http://localhost:3000")
METABASE_EMAIL = os.environ.get("METABASE_EMAIL", "admin@fraud.local")
METABASE_PASS = os.environ.get("METABASE_PASS", "Admin1234!")
DASHBOARDS_DIR = Path(__file__).parent.parent / "metabase" / "dashboards"


def _db_config() -> dict:
    _DB_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://fraud_user:dev_password@localhost:5432/fraud_db",
    )
    parsed = urlparse(_DB_URL)
    username = parsed.username or "fraud_user"
    password = parsed.password or "dev_password"
    hostname = parsed.hostname or "localhost"
    port = parsed.port or 5432
    dbname = parsed.path.lstrip("/") or "fraud_db"

    return {
        "engine": "postgres",
        "name": "Fraud Detection DB",
        "details": {
            "host": hostname,
            "port": int(port),
            "dbname": dbname,
            "user": username,
            "password": password,
            "ssl": False,
        },
    }


async def _authenticate(client: httpx.AsyncClient) -> str:
    response = await client.post(
        f"{METABASE_URL}/api/session",
        json={"username": METABASE_EMAIL, "password": METABASE_PASS},
    )
    response.raise_for_status()
    token = response.json()["id"]
    logger.info("Authenticated with Metabase.")
    return token


async def _get_or_create_db(client: httpx.AsyncClient, token: str) -> int:
    """Create the PostgreSQL connection in Metabase, or return existing ID."""
    headers = {"X-Metabase-Session": token}
    existing = await client.get(f"{METABASE_URL}/api/database", headers=headers)
    existing.raise_for_status()
    for db in existing.json().get("data", []):
        if db["name"] == "Fraud Detection DB":
            logger.info("Database connection already exists (id=%d).", db["id"])
            return db["id"]

    response = await client.post(
        f"{METABASE_URL}/api/database",
        headers=headers,
        json=_db_config(),
    )
    response.raise_for_status()
    db_id = response.json()["id"]
    logger.info("Created Metabase DB connection (id=%d).", db_id)
    return db_id


async def _sync_db(client: httpx.AsyncClient, token: str, db_id: int) -> None:
    headers = {"X-Metabase-Session": token}
    await client.post(
        f"{METABASE_URL}/api/database/{db_id}/sync_schema", headers=headers
    )
    logger.info("Database schema sync triggered.")


async def _get_or_create_card(
    client: httpx.AsyncClient, token: str, db_id: int, card_def: dict
) -> int:
    """Idempotently fetch or create a card (SQL query card) in Metabase."""
    headers = {"X-Metabase-Session": token}
    existing = await client.get(f"{METABASE_URL}/api/card", headers=headers)
    existing.raise_for_status()
    for card in existing.json():
        if card["name"] == card_def["name"]:
            logger.info("Card '%s' already exists (id=%d).", card_def["name"], card["id"])
            return card["id"]

    payload = {
        "name": card_def["name"],
        "dataset_query": {
            "database": db_id,
            "type": "native",
            "native": {
                "query": card_def["query"],
                "template-tags": {},
            },
        },
        "display": card_def["display"],
        "visualization_settings": {},
    }
    response = await client.post(
        f"{METABASE_URL}/api/card", headers=headers, json=payload
    )
    response.raise_for_status()
    card_id = response.json()["id"]
    logger.info("Created card '%s' (id=%d).", card_def["name"], card_id)
    return card_id


async def _import_dashboards(
    client: httpx.AsyncClient, token: str, db_id: int
) -> None:
    headers = {"X-Metabase-Session": token}
    dashboard_files = sorted(DASHBOARDS_DIR.glob("*.json"))
    if not dashboard_files:
        logger.warning("No dashboard JSON files found in %s", DASHBOARDS_DIR)
        return

    # Fetch existing dashboards
    existing_dashboards_resp = await client.get(
        f"{METABASE_URL}/api/dashboard", headers=headers
    )
    existing_dashboards_resp.raise_for_status()
    existing_dashboards = {d["name"]: d["id"] for d in existing_dashboards_resp.json()}

    for filepath in dashboard_files:
        dashboard = json.loads(filepath.read_text())
        dash_name = dashboard["name"]

        # 1. Get or create dashboard
        if dash_name in existing_dashboards:
            dash_id = existing_dashboards[dash_name]
            logger.info("Dashboard '%s' already exists (id=%d).", dash_name, dash_id)
        else:
            response = await client.post(
                f"{METABASE_URL}/api/dashboard",
                headers=headers,
                json={
                    "name": dash_name,
                    "description": dashboard.get("description", ""),
                },
            )
            response.raise_for_status()
            dash_id = response.json()["id"]
            logger.info("Created dashboard: %s (id=%d)", dash_name, dash_id)

        # Fetch dashboard details to check associated cards
        dash_detail_resp = await client.get(
            f"{METABASE_URL}/api/dashboard/{dash_id}", headers=headers
        )
        dash_detail_resp.raise_for_status()
        dash_detail = dash_detail_resp.json()
        existing_card_ids = {
            c["card_id"] for c in dash_detail.get("ordered_cards", []) if c.get("card_id")
        }

        # 2. Add SQL cards
        for card_def in dashboard.get("cards", []):
            card_id = await _get_or_create_card(client, token, db_id, card_def)
            if card_id not in existing_card_ids:
                # Add card to dashboard
                add_resp = await client.post(
                    f"{METABASE_URL}/api/dashboard/{dash_id}/cards",
                    headers=headers,
                    json={"cardId": card_id},
                )
                add_resp.raise_for_status()
                logger.info(
                    "Added card '%s' (id=%d) to dashboard '%s' (id=%d).",
                    card_def["name"],
                    card_id,
                    dash_name,
                    dash_id,
                )
            else:
                logger.info(
                    "Card '%s' (id=%d) already present on dashboard '%s'.",
                    card_def["name"],
                    card_id,
                    dash_name,
                )


async def main() -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        token = await _authenticate(client)
        db_id = await _get_or_create_db(client, token)
        await _sync_db(client, token, db_id)
        await _import_dashboards(client, token, db_id)
    logger.info("Metabase provisioning complete.")


if __name__ == "__main__":
    asyncio.run(main())
