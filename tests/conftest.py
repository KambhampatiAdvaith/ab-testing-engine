"""Pytest configuration — sets up an async in-memory SQLite database
so that the FastAPI endpoints that use SQLAlchemy can be tested without
a running PostgreSQL instance.
"""
import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.database import Base, get_db
from backend.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
_SessionLocal = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Override the database dependency with an in-memory SQLite database."""

    async def _create():
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _run(_create())

    async def override_get_db():
        async with _SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

    async def _drop():
        await _engine.dispose()

    _run(_drop())
