"""
Application settings — the single, immutable configuration root.

Assembles every configuration domain (app, llm, database, storage)
into one ``AppSettings`` class via **mixin inheritance**.  Each mixin
lives in its own file inside ``app.core.config`` and only declares the
fields that belong to its domain.

Why mixins instead of nested Pydantic models?
    To keep **full backward compatibility**: existing code accesses
    ``settings.llm_model``, not ``settings.llm.model``.  Mixins produce
    flat fields on the instance while keeping the *definition* organised
    by domain.

Environment resolution order (Pydantic Settings default):
    1. CLI arguments (not used here)
    2. Environment variables (highest priority among the three)
    3. ``.env`` file
    4. Field defaults (lowest priority)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config.app import ApplicationConfigMixin
from app.core.config.database import DatabaseConfigMixin
from app.core.config.llm import LLMConfigMixin
from app.core.config.storage import StorageConfigMixin


class AppSettings(
    ApplicationConfigMixin,
    LLMConfigMixin,
    DatabaseConfigMixin,
    StorageConfigMixin,
    BaseSettings,
):
    """Root settings — every configuration field is a flat attribute.

    The inheritance order matters for Pydantic field resolution:
    leftmost parent wins when two parents declare the same field name.
    ``BaseSettings`` is always the **last** (rightmost) parent so that
    mixin fields are properly recognised as settings fields.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # ``extra="ignore"`` makes the app tolerant of unknown env vars
        # (e.g. ``POSTGRES_PASSWORD`` injected by docker-compose).
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a process-wide cached ``AppSettings`` singleton.

    The ``lru_cache`` ensures ``.env`` is parsed exactly once,
    no matter how many modules call ``get_settings()``.
    """
    return AppSettings()
