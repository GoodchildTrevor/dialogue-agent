"""Unit tests for LangGraph edge functions.

after_router, after_orchestrator, after_tools are pure synchronous functions;
no mocking required.
"""
from __future__ import annotations

import pytest

from app.graph.edges import after_orchestrator, after_router, after_tools
from tests.conftest import base_state


# ---------------------------------------------------------------------------
# after_router
# ---------------------------------------------------------------------------

class TestAfterRouter:
    def test_final_answer_set_returns_end(self):
        state = base_state(final_answer="done")
        assert after_router(state) == "end"

    def test_next_action_respected_when_no_final_answer(self):
        state = base_state(final_answer="", next_action="reasoning")
        assert after_router(state) == "reasoning"

    def test_defaults_to_orchestrator_when_no_next_action(self):
        # next_action absent / empty string
        state = base_state(final_answer="", next_action="")
        # empty string is falsy so .get() returns it, but the fallback is "orchestrator"
        # Let's use a state without next_action key at all to hit the default
        minimal: dict = {"messages": [], "user_id": "", "request_id": ""}
        assert after_router(minimal) == "orchestrator"  # type: ignore[arg-type]

    def test_empty_final_answer_does_not_trigger_end(self):
        state = base_state(final_answer="", next_action="orchestrator")
        assert after_router(state) != "end"


# ---------------------------------------------------------------------------
# after_orchestrator
# ---------------------------------------------------------------------------

class TestAfterOrchestrator:
    def test_final_answer_set_returns_end(self):
        state = base_state(final_answer="result")
        assert after_orchestrator(state) == "end"

    def test_routes_to_next_action_when_set(self):
        state = base_state(final_answer="", next_action="tools")
        assert after_orchestrator(state) == "tools"

    def test_defaults_to_reasoning_when_no_next_action(self):
        minimal: dict = {"messages": [], "user_id": "", "request_id": ""}
        assert after_orchestrator(minimal) == "reasoning"  # type: ignore[arg-type]

    def test_empty_final_answer_does_not_trigger_end(self):
        state = base_state(final_answer="", next_action="reasoning")
        assert after_orchestrator(state) != "end"


# ---------------------------------------------------------------------------
# after_tools
# ---------------------------------------------------------------------------

class TestAfterTools:
    def test_routes_to_next_action(self):
        state = base_state(next_action="orchestrator")
        assert after_tools(state) == "orchestrator"

    def test_defaults_to_orchestrator_when_absent(self):
        minimal: dict = {"messages": [], "user_id": "", "request_id": ""}
        assert after_tools(minimal) == "orchestrator"  # type: ignore[arg-type]

    def test_custom_next_action_respected(self):
        state = base_state(next_action="reasoning")
        assert after_tools(state) == "reasoning"

    def test_final_answer_does_not_short_circuit(self):
        """after_tools does NOT check final_answer; it always follows next_action."""
        state = base_state(final_answer="done", next_action="orchestrator")
        assert after_tools(state) == "orchestrator"
