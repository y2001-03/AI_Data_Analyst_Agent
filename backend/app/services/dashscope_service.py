"""DashScope client service for dataset understanding."""

from __future__ import annotations

import json
from urllib import error, request

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.schemas.file import DatasetUploadResponse


class DashScopeService:
    """Minimal DashScope chat client."""

    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self) -> None:
        self.settings = get_settings()

    def analyze_dataset(self, file_info: DatasetUploadResponse) -> dict[str, object]:
        """Analyze dataset metadata with DashScope or fallback mock data."""
        api_key = self.settings.dashscope_api_key
        if not api_key:
            return self._mock_response(file_info)
        prompt = self._build_prompt(file_info)
        payload = self._build_payload(prompt)
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(api_key),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise AppException(f"DashScope request failed: {detail}", 502) from exc
        except error.URLError as exc:
            raise AppException("DashScope is unreachable.", 502) from exc
        return self._parse_response(body)

    def _build_payload(self, prompt: str) -> dict[str, object]:
        """Build the DashScope chat payload."""
        return {
            "model": self.settings.llm_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
        }

    def _build_headers(self, api_key: str) -> dict[str, str]:
        """Build request headers for DashScope."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _parse_response(self, body: str) -> dict[str, object]:
        """Parse the JSON payload returned by DashScope."""
        payload = json.loads(body)
        try:
            content = payload["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AppException("DashScope returned an invalid response.", 502) from exc
        return result

    def _build_prompt(self, file_info: DatasetUploadResponse) -> str:
        """Serialize dataset metadata into the model prompt."""
        payload = {
            "file_name": file_info.file_name,
            "file_type": file_info.file_type,
            "row_count": file_info.row_count,
            "column_count": file_info.column_count,
            "columns": [column.model_dump() for column in file_info.columns],
            "preview": file_info.preview,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _system_prompt(self) -> str:
        """Return the system prompt for dataset understanding."""
        return (
            "You are a data understanding assistant. "
            "Given dataset schema and preview rows, respond with JSON only. "
            "Use keys: summary and suggestions. "
            "summary must be a concise dataset description in Chinese. "
            "suggestions must be an array of 3 to 5 Chinese analysis suggestions."
        )

    def _mock_response(self, file_info: DatasetUploadResponse) -> dict[str, object]:
        """Return deterministic mock output when no API key is configured."""
        column_names = [column.name for column in file_info.columns[:3]]
        joined_names = ", ".join(column_names) if column_names else "unknown columns"
        summary = (
            f"This {file_info.file_type.upper()} dataset contains "
            f"{file_info.row_count} rows and {file_info.column_count} columns. "
            f"Key fields include {joined_names}."
        )
        suggestions = [
            "Check missing values and outliers in important columns.",
            "Analyze metric trends by time or category dimensions.",
            "Compare aggregated statistics across key business fields.",
        ]
        return {
            "summary": summary,
            "suggestions": suggestions,
        }
