"""Alembic environment configuration — async mode.

This env.py replaces the default synchronous alembic template with an
async-aware version that uses create_async_engine so that migrations
run against the same asyncpg driver used by the FastAPI application.
"""
import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Alembic Config object (gives access to .ini values)
config = context.config

# Logging setup from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# Import settings after logging is configured
from app.config import get_settings  # noqa: E402

settings = get_settings()

# target_metadata = None means Alembic uses raw SQL (no ORM models yet)
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() here emit the given string to the script output.
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live async database."""

    connectable = create_async_engine(settings.database_url, echo=False)

    async def _run() -> None:
        async with connectable.connect() as connection:
            await connection.run_sync(_do_run_migrations)
        await connectable.dispose()

    asyncio.run(_run())


def _do_run_migrations(connection: object) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
