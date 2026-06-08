"""Async SQLAlchemy engine and session factory.

Design decision reference: C-3 (use asyncpg, not psycopg2, to avoid blocking
the event loop under concurrent load).

Pool configuration:
- pool_size=20      — handles up to 20 concurrent DB sessions
- max_overflow=5    — allows burst of 5 additional connections
- pool_pre_ping=True — validates connections before use (detects stale connections)
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=5,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
