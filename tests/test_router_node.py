"""Unit tests for RouterNode.action routing logic.

Every test stubs the LLM client so no external service is required.
The trace() context-manager is patched to a no-op so there is no DB or
telemetry side-effect and no hanging async task.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import base_state, make_llm_client, _Settings
from app.graph.nodes.router import RouterNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(llm_response: str) -> RouterNode:
    return RouterNode(
        llm_client=make_llm_client(llm_response),
        settings=_Settings(),
    )


# Patch target: the `trace` context manager used inside RouterNode.action.
_TRACE_PATCH = "app.graph.nodes.router.trace"


def _noop_trace(*_args: object, **_kwargs: object):
    """Async context manager that does nothing."""
    return _NoopTraceCtx()


class _NoopTraceCtx:
    """Minimal async context-manager stub for `trace`."""

    async def __aenter__(self):
        obj = MagicMock()
        # Allow arbitrary attribute assignment without raising
        return obj

    async def __aexit__(self, *_):
        return False


# ---------------------------------------------------------------------------
# Test: simple query → final_answer set, next_action == "end"
# ---------------------------------------------------------------------------

class TestSimpleQuery:
    async def test_returns_final_answer(self):
        payload = json.dumps({"is_simple": True, "answer": "42"})
        router = _make_router(payload)
        state = base_state(messages=[{"role": "user", "content": "What is 6*7?"}])

        with patch(_TRACE_PATCH, side_effect=_noop_trace):
            result = await router.action(state)

        assert result["final_answer"] == "42"
        assert result["next_action"] == "end"

    async def test_is_complex_task_not_set_for_simple(self):
        payload = json.dumps({"is_simple": True, "answer": "Paris"})
        router = _make_router(payload)
        state = base_state(messages=[{"role": "user", "content": "Capital of France?"}])

        with patch(_TRACE_PATCH, side_effect=_noop_trace):
            result = await router.action(state)

        assert "is_complex_task" not in result


# ---------------------------------------------------------------------------
# Test: complex query → is_complex_task=True, next_action == "reasoning"
# ---------------------------------------------------------------------------

class TestComplexQuery:
    @pytest.mark.parametrize(
        "flag_key",
        ["needs_reasoning_model", "is_complex_task"],
    )
    async def test_routes_to_reasoning(self, flag_key: str):
        payload = json.dumps({flag_key: True})
        router = _make_router(payload)
        state = base_state(
            messages=[{"role": "user", "content": "Explain quantum entanglement in depth"}]
        )

        with patch(_TRACE_PATCH, side_effect=_noop_trace):
            result = await router.action(state)

        assert result["is_complex_task"] is True
        assert result["next_action"] == "reasoning"

    async def test_no_final_answer_on_complex(self):
        payload = json.dumps({"needs_reasoning_model": True})
        router = _make_router(payload)
        state = base_state(
            messages=[{"role": "user", "content": "Prove Riemann hypothesis"}]
        )

        with patch(_TRACE_PATCH, side_effect=_noop_trace):
            result = await router.action(state)

        assert "final_answer" not in result


# ---------------------------------------------------------------------------
# Test: tool / orchestrator query → next_action == "orchestrator"
# ---------------------------------------------------------------------------

class TestToolQuery:
    async def test_routes_to_orchestrator(self):
        payload = json.dumps({"is_simple": False, "is_complex_task": False})
        router = _make_router(payload)
        state = base_state(
            messages=[{"role": "user", "content": "Search the web for latest AI news"}]
        )

        with patch(_TRACE_PATCH, side_effect=_noop_trace):
            result = await router.action(state)

        assert result["next_action"] == "orchestrator"
        assert result["is_complex_task"] is False

    async def test_no_final_answer_on_tool_query(self):
        payload = json.dumps({"is_simple": False})
        router = _make_router(payload)
        state = base_state(
            messages=[{"role": "user", "content": "Book a flight to Paris"}]
        )

        with patch(_TRACE_PATCH, side_effect=_noop_trace):
            result = await router.action(state)

        assert "final_answer" not in result


# ---------------------------------------------------------------------------
# Test: fallback (LLM returns unparseable JSON) → orchestrator via _router_fallback
# ---------------------------------------------------------------------------

class TestFallback:
    async def test_invalid_json_falls_back_to_orchestrator(self):
        router = _make_router("NOT JSON AT ALL")
        state = base_state(
            messages=[{"role": "user", "content": "Do something"}]
        )

        with patch(_TRACE_PATCH, side_effect=_noop_trace):
            result = await router.action(state)

        # _router_fallback always sets next_action to "orchestrator"
        assert result["next_action"] == "orchestrator"
