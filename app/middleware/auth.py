"""API key authentication middleware.

Design decision reference: C-1
- All endpoints require X-API-Key header (header name configurable via settings).
- Keys are stored as SHA-256 hashes in the api_keys PostgreSQL table.
- Both /predict and /stream return HTTP 401 (not 403) to avoid endpoint enumeration.
- /health, /docs, /openapi.json, /redoc are explicitly whitelisted (unauthenticated).
- Rate limiting (100 req/s per key) is enforced via slowapi in Phase 2.

Note on BaseHTTPMiddleware + HTTPException:
    BaseHTTPMiddleware does not propagate FastAPI's HTTPException to the
    exception handlers; instead, raise HTTPException causes the exception to
    surface to the ASGI transport layer in tests. We return JSONResponse
    directly from the middleware to avoid this behaviour and produce correct
    HTTP 401 responses in all environments.
"""
import hashlib

import sqlalchemy as sa
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings
from app.db.engine import AsyncSessionLocal

settings = get_settings()

# Paths that bypass API key authentication
UNAUTHENTICATED_PATHS: frozenset[str] = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces API key authentication.

    Flow:
        1. If path is in UNAUTHENTICATED_PATHS — pass through.
        2. Extract X-API-Key header (name from settings).
        3. SHA-256 hash the key and look up in api_keys table.
        4. If not found or inactive — raise HTTP 401.
        5. Attach key_id to request state for downstream logging.
    """

    async def dispatch(self, request: Request, call_next: type) -> Response:
        if request.url.path in UNAUTHENTICATED_PATHS:
            return await call_next(request)

        api_key: str | None = request.headers.get(settings.api_key_header)
        if not api_key:
            return JSONResponse(
                status_code=401, content={"detail": "Missing API key"}
            )

        key_hash: str = hashlib.sha256(api_key.encode()).hexdigest()

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sa.text(
                    "SELECT id FROM api_keys WHERE key_hash = :hash AND is_active = true"
                ),
                {"hash": key_hash},
            )
            row = result.fetchone()

        if not row:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or inactive API key"}
            )

        # Attach key_id to request state for audit logging in Phase 2
        request.state.api_key_id = str(row[0])
        return await call_next(request)
