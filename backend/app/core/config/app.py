"""
Application-level configuration.

Defines the fields that describe the running application itself:
name, environment, debug mode, network bind address, and port.

This is a **mixin** — it only declares fields.  The actual
``pydantic_settings.BaseSettings`` subclass is assembled in
``app.core.config.settings.AppSettings``.
"""

from pydantic import Field


class ApplicationConfigMixin:
    """Application identity and runtime-mode fields.

    These fields are merged flat into ``AppSettings`` via multiple
    inheritance, so existing code that accesses ``settings.app_name``
    (or any other field declared here) continues to work without changes.
    """

    app_name: str = Field(
        default="AI Data Analyst Agent",
        alias="APP_NAME",
        description="Human-readable application name shown in docs and logs.",
    )

    app_env: str = Field(
        default="development",
        alias="APP_ENV",
        description="Runtime environment: development | testing | production.",
    )

    app_debug: bool = Field(
        default=True,
        alias="APP_DEBUG",
        description="Enable FastAPI debug mode and verbose logging.",
    )

    app_host: str = Field(
        default="0.0.0.0",
        alias="APP_HOST",
        description="Network interface the ASGI server binds to.",
    )

    app_port: int = Field(
        default=8000,
        alias="APP_PORT",
        description="TCP port the ASGI server listens on.",
    )
