from __future__ import annotations

from app.graph.state import AssistantState


def after_router(state: AssistantState) -> str:
    if state.get("final_answer"):
        return "end"
    return state.get("next_action", "orchestrator")


def after_orchestrator(state: AssistantState) -> str:
    if state.get("final_answer"):
        return "end"
    return state.get("next_action", "reasoning")


def after_tools(state: AssistantState) -> str:
    """Escalate to reasoning model after 3 consecutive tool failures; otherwise retry orchestrator."""
    if state.get("last_tool_error") and state.get("tool_retry_count", 0) >= 3:
        return "reasoning"
    return "orchestrator"
