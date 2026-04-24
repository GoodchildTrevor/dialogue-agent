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
    return state.get("next_action", "orchestrator")

