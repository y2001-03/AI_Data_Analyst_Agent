"""FastAPI dependencies."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.models.base import SessionLocal


def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

