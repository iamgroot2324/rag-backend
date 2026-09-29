"""Shared pytest fixtures and environment setup."""

import os

# Must be set before any app module is imported: settings are read at import time,
# and tests use an in-memory database instead of the real one.
os.environ.setdefault("DATABASE_URL", "sqlite://")

from collections.abc import Iterator
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401  (registers tables on Base)
from app.db.session import Base


@pytest.fixture
def db() -> Iterator[Session]:
    """A fresh in-memory SQLite database per test."""
    # StaticPool keeps a single shared connection so the in-memory DB isn't lost
    # between sessions/threads.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def future_date() -> str:
    """A valid interview date (7 days from today) as YYYY-MM-DD."""
    return (date.today() + timedelta(days=7)).isoformat()
