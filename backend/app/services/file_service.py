"""File ingestion service."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

from app.core.exceptions import AppException
from app.schemas.file import DatasetColumnProfile, DatasetUploadResponse


class FileService:
    """Handle dataset validation, parsing, and profiling."""

    allowed_extensions = {".csv", ".xlsx"}
    preview_rows = 5

    async def parse_upload(self, file: UploadFile) -> DatasetUploadResponse:
        """Parse a CSV or XLSX upload into structured dataset metadata."""
        file_name = file.filename or "unknown"
        extension = Path(file_name).suffix.lower()
        self._validate_extension(extension)
        content = await file.read()
        if not content:
            raise AppException("Uploaded file is empty.", status_code=400)
        dataframe = self._load_dataframe(content, extension)
        dataframe = dataframe.where(pd.notnull(dataframe), None)
        schema = self._build_schema(dataframe)
        preview = self._build_preview(dataframe)
        return DatasetUploadResponse(
            file_name=file_name,
            file_type=extension.replace(".", ""),
            row_count=len(dataframe.index),
            column_count=len(dataframe.columns),
            preview=preview,
            columns=schema,
        )

    def _validate_extension(self, extension: str) -> None:
        """Validate that the uploaded file type is supported."""
        if extension not in self.allowed_extensions:
            raise AppException(
                "Unsupported file type. Please upload a CSV or XLSX file.",
                status_code=400,
            )

    def _load_dataframe(self, content: bytes, extension: str) -> pd.DataFrame:
        """Load the file content into a pandas DataFrame."""
        file_buffer = BytesIO(content)
        try:
            if extension == ".csv":
                return pd.read_csv(file_buffer)
            return pd.read_excel(file_buffer)
        except Exception as exc:
            raise AppException(
                "Unable to parse the uploaded file.",
                status_code=400,
            ) from exc

    def _build_schema(self, dataframe: pd.DataFrame) -> list[DatasetColumnProfile]:
        """Infer column profiles from a DataFrame."""
        schema: list[DatasetColumnProfile] = []
        for column in dataframe.columns:
            series = dataframe[column]
            schema.append(
                DatasetColumnProfile(
                    name=str(column),
                    data_type=str(series.dtype),
                    missing_count=int(series.isnull().sum()),
                    unique_values=int(series.nunique(dropna=True)),
                )
            )
        return schema

    def _build_preview(
        self,
        dataframe: pd.DataFrame,
    ) -> list[dict[str, str | int | float | bool | None]]:
        """Convert the first rows of a DataFrame into JSON-safe records."""
        preview_frame = dataframe.head(self.preview_rows)
        records = preview_frame.to_dict(orient="records")
        return [self._normalize_record(record) for record in records]

    def _normalize_record(
        self,
        record: dict[str, Any],
    ) -> dict[str, str | int | float | bool | None]:
        """Normalize pandas values into JSON-friendly primitives."""
        normalized: dict[str, str | int | float | bool | None] = {}
        for key, value in record.items():
            normalized[str(key)] = self._normalize_value(value)
        return normalized

    def _normalize_value(self, value: Any) -> str | int | float | bool | None:
        """Normalize a scalar value for API responses."""
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
