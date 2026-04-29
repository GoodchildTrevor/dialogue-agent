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
def _make_state(
    content: str = "Hello",
    user_id: str = "u1",
    request_id: str = "r1",
) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": content}],
        "user_id": user_id,
        "request_id": request_id,
    }


def _make_router(
    llm_response_content: str,
    router_model: str = "test-model",
) -> RouterNode:
    """Build a RouterNode with a fully mocked LLM client and trace context."""
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(
        return_value={
            "message": {"content": llm_response_content},
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
    )

    settings = MagicMock()
    settings.ROUTER_MODEL = router_model

    return RouterNode(llm_client=llm_client, settings=settings)


# ---------------------------------------------------------------------------
# Patch trace so it does not require a real DB / tracing backend
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_trace(monkeypatch):
    """Replace app.graph.nodes.router.trace with a no-op async context manager."""
    import contextlib

    @contextlib.asynccontextmanager
    async def _noop_trace(**kwargs):
        mock_span = MagicMock()
        mock_span.input_hash = None
        mock_span.estimated_tokens = None
        mock_span.model_used = None
        mock_span.output = None
        mock_span.route_decision = None
        yield mock_span

    monkeypatch.setattr("app.graph.nodes.router.trace", _noop_trace)


# ---------------------------------------------------------------------------
# Tests: simple query  →  end
# ---------------------------------------------------------------------------

class TestRouterSimpleQuery:
    """When the LLM marks the query as simple (is_simple + answer present),
    router_node must return final_answer and next_action='end'."""

    @pytest.mark.asyncio
    async def test_simple_query_returns_final_answer(self):
        router = _make_router('{"is_simple": true, "answer": "42"}')
        result = await router.action(_make_state("What is 6x7?"))

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
    @pytest.mark.asyncio
    async def test_simple_query_no_is_complex_task(self):
        router = _make_router('{"is_simple": true, "answer": "Paris"}')
        result = await router.action(_make_state("Capital of France?"))

        assert "is_complex_task" not in result or not result.get("is_complex_task")


# ---------------------------------------------------------------------------
# Tests: complex query  →  reasoning
# ---------------------------------------------------------------------------

class TestRouterComplexQuery:
    """When the LLM marks the query as complex/needing reasoning,
    next_action must be 'reasoning' and is_complex_task must be True."""

    @pytest.mark.asyncio
    async def test_needs_reasoning_model(self):
        router = _make_router('{"needs_reasoning_model": true}')
        result = await router.action(_make_state("Prove the Riemann hypothesis."))

        assert result["next_action"] == "reasoning"
        assert result["is_complex_task"] is True

    @pytest.mark.asyncio
    async def test_is_complex_task_flag(self):
        router = _make_router('{"is_complex_task": true}')
        result = await router.action(_make_state("Design a distributed database."))

        assert result["next_action"] == "reasoning"
        assert result["is_complex_task"] is True


# ---------------------------------------------------------------------------
# Tests: tool / orchestrator query  →  orchestrator
# ---------------------------------------------------------------------------

class TestRouterOrchestratorQuery:
    """For non-simple, non-complex queries the router must route to orchestrator."""

    @pytest.mark.asyncio
    async def test_tool_query_routes_to_orchestrator(self):
        router = _make_router('{"is_complex_task": false}')
        result = await router.action(_make_state("Search the web for Python 3.13 changes."))

        assert result["next_action"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_no_final_answer_on_orchestrator_route(self):
        router = _make_router('{"is_complex_task": false}')
        result = await router.action(_make_state("Run a tool."))

        assert "final_answer" not in result or result.get("final_answer") is None

    @pytest.mark.asyncio
    async def test_invalid_json_fallback_routes_to_orchestrator(self):
        """Unparseable LLM response falls back via _router_fallback → orchestrator."""
        router = _make_router("not json at all")
        result = await router.action(_make_state("Tell me something."))

        # _router_fallback returns {"is_complex_task": False} for unknown messages
        assert result["next_action"] in ("orchestrator", "reasoning", "end")
