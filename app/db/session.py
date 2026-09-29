"""SQLAlchemy engine, session factory and declarative base."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

_settings = get_settings()

# SQLite only allows a connection to be used by the thread that created it;
# FastAPI runs sync endpoints in a thread pool, so we relax that check for SQLite.
_connect_args = {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}

engine = create_engine(_settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class all ORM models inherit from."""


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yield a session per request and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
