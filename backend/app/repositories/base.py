"""Repository base classes."""

from sqlalchemy.orm import Session


class BaseRepository:
    """Base repository wrapper."""

    def __init__(self, session: Session) -> None:
        self.session = session

