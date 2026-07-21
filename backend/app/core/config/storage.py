"""
File-storage configuration.

Covers uploaded dataset handling: where files are stored, maximum
allowed size, and allowed extensions.

This is a **mixin** — see ``app.core.config.app`` for the pattern
rationale.
"""

from pydantic import Field


class StorageConfigMixin:
    """File-upload and storage fields merged flat into ``AppSettings``.

    .. note::

        These fields are **forward-looking** — they are not yet consumed
        by any service (the current implementation processes uploads
        in memory).  They are defined here so that future storage
        features (local disk, S3, etc.) have a canonical configuration
        source from day one.
    """

    upload_dir: str = Field(
        default="./uploads",
        alias="UPLOAD_DIR",
        description="Local directory for persisting uploaded datasets.",
    )

    upload_max_size_mb: int = Field(
        default=10,
        alias="UPLOAD_MAX_SIZE_MB",
        description="Maximum allowed upload file size in megabytes.",
    )
