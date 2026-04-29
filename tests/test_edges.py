"""Unit tests for LangGraph edge functions (after_router, after_orchestrator, after_tools)."""
from __future__ import annotations

import pytest

from app.graph.edges import after_orchestrator, after_router, after_tools


# ---------------------------------------------------------------------------
# after_router
# ---------------------------------------------------------------------------

class TestAfterRouter:
    def test_final_answer_returns_end(self):
        state = {"final_answer": "Done", "next_action": "orchestrator"}
        assert after_router(state) == "end"

    def test_no_final_answer_returns_next_action(self):
        state = {"next_action": "orchestrator"}
        assert after_router(state) == "orchestrator"

    def test_no_final_answer_reasoning_next_action(self):
        state = {"next_action": "reasoning"}
        assert after_router(state) == "reasoning"

    def test_empty_state_defaults_to_orchestrator(self):
        """If neither final_answer nor next_action is set, default is 'orchestrator'."""
        assert after_router({}) == "orchestrator"

    def test_empty_final_answer_string_is_falsy_continue(self):
        """An empty string for final_answer must NOT route to end."""
        state = {"final_answer": "", "next_action": "orchestrator"}
        assert after_router(state) == "orchestrator"


# ---------------------------------------------------------------------------
# after_orchestrator
# ---------------------------------------------------------------------------

class TestAfterOrchestrator:
    def test_final_answer_returns_end(self):
        state = {"final_answer": "All done.", "next_action": "reasoning"}
        assert after_orchestrator(state) == "end"

    def test_no_final_answer_returns_next_action(self):
        state = {"next_action": "tools"}
        assert after_orchestrator(state) == "tools"

    def test_default_fallback_is_reasoning(self):
        """Without next_action, default fallback is 'reasoning'."""
        assert after_orchestrator({}) == "reasoning"

    def test_empty_final_answer_continues(self):
        state = {"final_answer": "", "next_action": "tools"}
        assert after_orchestrator(state) == "tools"


# ---------------------------------------------------------------------------
# after_tools
# ---------------------------------------------------------------------------

class TestAfterTools:
    def test_returns_next_action(self):
        state = {"next_action": "orchestrator"}
        assert after_tools(state) == "orchestrator"

    def test_default_fallback_is_orchestrator(self):
        """Without next_action, after_tools defaults to 'orchestrator'."""
        assert after_tools({}) == "orchestrator"

    def test_arbitrary_next_action_is_forwarded(self):
        state = {"next_action": "some_other_node"}
        assert after_tools(state) == "some_other_node"
