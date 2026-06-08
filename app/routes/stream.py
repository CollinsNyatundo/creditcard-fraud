"""WebSocket /stream endpoint and stream token generation. Resolution: S-4, C-1, U-4."""
import asyncio
import json
import secrets
from datetime import datetime, timezone
import redis.asyncio as aioredis
import sqlalchemy as sa
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Request, HTTPException
from app.config import get_settings
from app.db.engine import AsyncSessionLocal
from app.services.redis_cache import get_redis

router = APIRouter()
settings = get_settings()

STREAM_CHANNEL = "fraud:stream"


@router.post("/auth/stream-token")
async def generate_stream_token(request: Request) -> dict[str, str]:
    """Generate a short-lived token for WebSocket stream authentication."""
    api_key_id: str | None = getattr(request.state, "api_key_id", None)
    if not api_key_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = secrets.token_urlsafe(32)
    redis: aioredis.Redis = await get_redis()
    token_key = f"stream_token:{token}"
    await redis.setex(token_key, 60, api_key_id)

    return {"token": token}


async def _replay_from_db(
    websocket: WebSocket,
    since: datetime | None,
) -> None:
    """Fetch historical rows from PostgreSQL and send as replay burst."""
    if since is None:
        cutoff = datetime.now(timezone.utc).timestamp() - settings.stream_default_replay_seconds
        since = datetime.fromtimestamp(cutoff, tz=timezone.utc)

    # Enforce maximum replay limit of 10 minutes (600 seconds)
    max_cutoff = datetime.now(timezone.utc).timestamp() - min(settings.stream_max_replay_seconds, 600)
    max_since = datetime.fromtimestamp(max_cutoff, tz=timezone.utc)
    if since < max_since:
        since = max_since  # cap replay window to prevent overloading

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa.text("""
                SELECT id::text, card_id, amount, fraud_probability,
                       is_flagged, latency_ms, created_at
                FROM predictions
                WHERE created_at >= :since
                ORDER BY created_at ASC
                LIMIT 500
            """),
            {"since": since},
        )
        rows = result.fetchall()

    for row in rows:
        await websocket.send_text(json.dumps({
            "type": "replay",
            "prediction_id": row[0],
            "card_id": row[1],
            "amount": float(row[2]),
            "fraud_probability": row[3],
            "is_flagged": row[4],
            "latency_ms": row[5],
            "created_at": row[6].isoformat(),
        }))


@router.websocket("/stream")
async def stream(
    websocket: WebSocket,
    token: str | None = Query(default=None),
    since: str | None = Query(default=None),
) -> None:
    """WebSocket live fraud stream.

    Requires a short-lived token generated via /auth/stream-token.
    Optional query param: ?since=<ISO-8601 timestamp> for replay.
    """
    await websocket.accept()

    if not token:
        await websocket.send_text(json.dumps({"error": "Missing stream token"}))
        await websocket.close(code=1008)
        return

    redis: aioredis.Redis = await get_redis()
    token_key = f"stream_token:{token}"
    api_key_id = await redis.get(token_key)

    if not api_key_id:
        await websocket.send_text(json.dumps({"error": "Invalid or expired stream token"}))
        await websocket.close(code=1008)
        return

    # Delete token immediately to make it single-use
    await redis.delete(token_key)

    # Step 1: Replay catch-up
    since_dt: datetime | None = None
    if since:
        try:
            # Normalize Z suffix to +00:00 for python fromisoformat
            norm_since = since.replace("Z", "+00:00") if since.endswith("Z") else since
            since_dt = datetime.fromisoformat(norm_since)
        except ValueError:
            await websocket.send_text(json.dumps({"error": "invalid since format"}))
            await websocket.close(code=1003)
            return

    try:
        await _replay_from_db(websocket, since_dt)
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": f"Replay failed: {str(e)}"}))
            await websocket.close(code=1011)
        except RuntimeError:
            pass
        return

    # Step 2: Switch to live Pub/Sub
    pubsub = redis.pubsub()
    await pubsub.subscribe(STREAM_CHANNEL)

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            payload = json.loads(message["data"])

            # Staleness check — drop messages older than threshold
            created_at = datetime.fromisoformat(payload["created_at"])
            age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age_seconds > settings.stream_stale_threshold_seconds:
                continue

            await websocket.send_text(json.dumps({**payload, "type": "live"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": f"Stream error: {str(e)}"}))
        except RuntimeError:
            pass
    finally:
        await pubsub.unsubscribe(STREAM_CHANNEL)
        await pubsub.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass
