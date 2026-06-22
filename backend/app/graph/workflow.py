"""LangGraph workflow assembly."""

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    analyze_intent,
    generate_report,
    load_memory,
    plan_task,
    run_analysis_tool,
    run_visualization_tool,
    save_memory,
    understand_data,
)
from app.agents.state import AgentState


def build_analysis_graph():
    """Build the LangGraph workflow for the analysis agent."""
    graph = StateGraph(AgentState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("analyze_intent", analyze_intent)
    graph.add_node("understand_data", understand_data)
    graph.add_node("plan_task", plan_task)
    graph.add_node("run_analysis_tool", run_analysis_tool)
    graph.add_node("run_visualization_tool", run_visualization_tool)
    graph.add_node("generate_report", generate_report)
    graph.add_node("save_memory", save_memory)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "analyze_intent")
    graph.add_edge("analyze_intent", "understand_data")
    graph.add_edge("understand_data", "plan_task")
    graph.add_edge("plan_task", "run_analysis_tool")
    graph.add_edge("run_analysis_tool", "run_visualization_tool")
    graph.add_edge("run_visualization_tool", "generate_report")
    graph.add_edge("generate_report", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()

