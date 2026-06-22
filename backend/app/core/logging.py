"""Logging configuration."""

import logging


def configure_logging(environment: str) -> None:
    """Configure application-wide logging."""
    level = logging.DEBUG if environment == "development" else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

