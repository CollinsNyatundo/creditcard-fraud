"""Runtime config loaded from system_config table with in-memory TTL cache. Resolution: C-4."""
import asyncio
import time
import sqlalchemy as sa
from app.db.engine import AsyncSessionLocal


class ConfigService:
    def __init__(self, ttl: int = 60) -> None:
        self._cache: dict[str, tuple[object, float]] = {}
        self._ttl = ttl
        self._lock: asyncio.Lock | None = None

    async def get(self, key: str, default: object = None) -> object:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.monotonic() < expires_at:
                    return value

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    sa.text("SELECT value FROM system_config WHERE key = :key"),
                    {"key": key},
                )
                row = result.fetchone()

            value = row[0] if row else default
            self._cache[key] = (value, time.monotonic() + self._ttl)
            return value

    async def get_float(self, key: str, default: float) -> float:
        raw = await self.get(key, str(default))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default


# Module-level singleton
config_service = ConfigService(ttl=60)
