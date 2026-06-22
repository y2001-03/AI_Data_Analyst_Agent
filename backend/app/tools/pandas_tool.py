"""Pandas analysis tool placeholder."""

from dataclasses import dataclass


@dataclass
class AnalysisToolResult:
    """Structured output for the analysis tool."""

    summary: str


class PandasAnalysisTool:
    """Placeholder Pandas-based analysis tool."""

    def run(self, question: str) -> AnalysisToolResult:
        """Return a placeholder response for analysis execution."""
        return AnalysisToolResult(
            summary=f"Pandas analysis placeholder executed for: {question}"
        )

