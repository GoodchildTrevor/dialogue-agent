from __future__ import annotations

import logging

from app.graph.state import AssistantState

log = logging.getLogger(__name__)


def after_router(state: AssistantState) -> str:
    if state.get("final_answer"):
        destination = "end"
    else:
        destination = state.get("next_action", "orchestrator")
    log.debug(
        "[%s] after_router -> %s",
        state.get("request_id", "?"),
        destination,
    )
    return destination


def after_orchestrator(state: AssistantState) -> str:
    if state.get("final_answer"):
        destination = "end"
    else:
        destination = state.get("next_action", "reasoning")
    log.debug(
        "[%s] after_orchestrator -> %s (final_answer=%r, next_action=%r)",
        state.get("request_id", "?"),
        destination,
        bool(state.get("final_answer")),
        state.get("next_action"),
    )
    return destination


def after_tools(state: AssistantState) -> str:
    destination = state.get("next_action", "orchestrator")
    log.debug(
        "[%s] after_tools -> %s (retry_count=%d)",
        state.get("request_id", "?"),
        destination,
        state.get("tool_retry_count", 0),
    )
    return destination
