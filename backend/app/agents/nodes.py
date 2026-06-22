"""Workflow node implementations."""

from app.agents.state import AgentState
from app.memory.service import MemoryService
from app.rag.service import RAGService
from app.tools.pandas_tool import PandasAnalysisTool
from app.tools.visualization_tool import VisualizationTool


memory_service = MemoryService()
rag_service = RAGService()
analysis_tool = PandasAnalysisTool()
visualization_tool = VisualizationTool()


def load_memory(state: AgentState) -> AgentState:
    """Load user preferences into the workflow state."""
    state["steps"].append("load_memory")
    state["messages"].append(memory_service.load_context(state["session_id"]))
    return state


def analyze_intent(state: AgentState) -> AgentState:
    """Infer a placeholder intent for the user question."""
    state["steps"].append("intent_analysis")
    state["intent"] = "general_analysis"
    return state


def understand_data(state: AgentState) -> AgentState:
    """Populate a placeholder data summary."""
    state["steps"].append("data_understanding")
    state["data_summary"] = {"status": "pending_file_parsing"}
    return state


def plan_task(state: AgentState) -> AgentState:
    """Create a placeholder execution plan."""
    state["steps"].append("task_planning")
    state["plan"] = [
        "inspect schema",
        "compute metrics",
        "generate chart",
        "write report",
    ]
    return state


def run_analysis_tool(state: AgentState) -> AgentState:
    """Run the placeholder Pandas analysis tool."""
    state["steps"].append("python_analysis_tool")
    result = analysis_tool.run(question=state["question"])
    state["messages"].append(result.summary)
    return state


def run_visualization_tool(state: AgentState) -> AgentState:
    """Run the placeholder visualization tool."""
    state["steps"].append("visualization_tool")
    chart = visualization_tool.run(question=state["question"])
    state["charts"].append(chart)
    return state


def generate_report(state: AgentState) -> AgentState:
    """Generate a placeholder markdown report."""
    state["steps"].append("report_generator")
    references = rag_service.retrieve(state["question"])
    state["report_markdown"] = "\n".join(
        [
            "# Analysis Report",
            "",
            f"- Question: {state['question']}",
            f"- Intent: {state['intent']}",
            f"- References: {', '.join(references)}",
        ]
    )
    state["final_answer"] = "Analysis workflow initialized. Business logic is pending."
    return state


def save_memory(state: AgentState) -> AgentState:
    """Persist placeholder memory after analysis."""
    state["steps"].append("save_memory")
    memory_service.save_context(state["session_id"], state["question"])
    return state

