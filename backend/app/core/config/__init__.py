"""
Configuration centre — unified, domain-grouped, environment-aware.

Public API (backward-compatible with the previous single-file layout)::

    from app.core.config import get_settings      # ← most common
    from app.core.config import AppSettings        # class reference
    from app.core.config import Settings           # legacy alias

    settings = get_settings()
    # All fields are flat — existing code continues to work:
    settings.llm_model          # from LLMConfigMixin
    settings.database_url       # from DatabaseConfigMixin
    settings.app_name           # from ApplicationConfigMixin

For typed access to a specific domain::

    from app.core.config.llm import LLMConfigMixin
    from app.core.config.database import DatabaseConfigMixin
    # … etc.
"""

from app.core.config.settings import AppSettings, get_settings

# Legacy alias — retained in case any external code was importing
# ``Settings`` by name (the original class name before the refactor).
Settings = AppSettings

__all__ = [
    "AppSettings",
    "Settings",
    "get_settings",
]
