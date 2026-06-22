"""Tool registry for deterministic execution tools."""

from __future__ import annotations

from app.tools.dataframe_tools import DataframeTool, GroupByTool, StatsTool, TrendTool
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

    @classmethod
    def with_default_tools(cls) -> "ToolRegistry":
        """Build a registry populated with the standard pandas tools."""
        registry = cls()
        registry.register(GroupByTool())
        registry.register(StatsTool())
        registry.register(TrendTool())
        registry.register(SQLTool())
        return registry
