"""Tool registry for deterministic execution tools."""

from __future__ import annotations

from app.schemas.tool_capability import ToolCapability
from app.tools.anomaly_detection_tool import AnomalyDetectionTool
from app.tools.capabilities import validate_capability
from app.tools.correlation_analysis_tool import CorrelationAnalysisTool
from app.tools.data_cleaning_tool import DataCleaningTool
from app.tools.data_quality_tool import DataQualityTool
from app.tools.dataframe_tools import DataframeTool, GroupByTool, StatsTool, TrendTool
from app.tools.distribution_analysis_tool import DistributionAnalysisTool
from app.tools.forecast_analysis_tool import ForecastAnalysisTool
from app.tools.sql_tool import SQLTool


class ToolRegistry:
    """Register and resolve execution tools by name."""

    def __init__(self) -> None:
        self._tools: dict[str, DataframeTool] = {}

    def register(self, tool: DataframeTool) -> None:
        """Register a tool instance under its stable name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> DataframeTool | None:
        """Return a registered tool by name."""
        return self._tools.get(name)

    def list_capabilities(self) -> list[ToolCapability]:
        """Return validated capabilities in stable tool-name order."""
        return [validate_capability(self._tools[name]) for name in sorted(self._tools)]

    def get_capability(self, tool_name: str) -> ToolCapability | None:
        """Return one validated capability, following get() semantics for unknown names."""
        tool = self.get(tool_name)
        return validate_capability(tool) if tool is not None else None

    def find_by_task_type(self, task_type: str) -> list[ToolCapability]:
        """Find exact task-type matches without keyword inference."""
        return [capability for capability in self.list_capabilities() if task_type in capability.task_types]

    def find_capabilities_by_category(self, category: str) -> list[ToolCapability]:
        """Find exact category matches in stable order."""
        return [capability for capability in self.list_capabilities() if capability.category == category]

    def get_capability_catalog(self) -> dict[str, object]:
        """Build a deterministic JSON-safe catalog without executing registered tools."""
        capabilities = self.list_capabilities()
        task_type_index: dict[str, list[str]] = {}
        category_index: dict[str, list[str]] = {}
        for capability in capabilities:
            for task_type in capability.task_types:
                task_type_index.setdefault(task_type, []).append(capability.tool_name)
            category_index.setdefault(capability.category, []).append(capability.tool_name)
        return {
            "tools": [capability.model_dump(mode="json") for capability in capabilities],
            "task_type_index": {
                key: sorted(value) for key, value in sorted(task_type_index.items())
            },
            "category_index": {
                key: sorted(value) for key, value in sorted(category_index.items())
            },
            "metadata": {
                "tool_count": len(capabilities),
                "category_count": len(category_index),
                "stable_sort": "tool_name_ascending",
            },
        }

    @classmethod
    def with_default_tools(cls) -> "ToolRegistry":
        """Build a registry populated with the standard pandas tools."""
        registry = cls()
        registry.register(GroupByTool())
        registry.register(StatsTool())
        registry.register(TrendTool())
        registry.register(AnomalyDetectionTool())
        registry.register(CorrelationAnalysisTool())
        registry.register(DistributionAnalysisTool())
        registry.register(ForecastAnalysisTool())
        registry.register(DataQualityTool())
        registry.register(DataCleaningTool())
        registry.register(SQLTool())
        return registry
