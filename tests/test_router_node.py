"""Tests for the router node."""
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.router import RouterNode


def _make_state(**overrides) -> dict[str, Any]:
    """Create a base state dict with optional overrides."""
    state = {
        "request_id": "test-request-id",
        "user_id": "test-user-id",
        "messages": [{"role": "user", "content": "test message"}],
        "is_complex_task": False,
    }
    state.update(overrides)
    return state


def _make_router(llm_response: dict) -> RouterNode:
    """Create a router node with a mock LLM client."""
    client = MagicMock()
    client.chat.return_value = llm_response
    settings = MagicMock()
    settings.ROUTER_MODEL = "test-model"
    settings.REASONING_MODEL = "test-reasoning-model"
    settings.MAX_TOKENS = 1000
    return RouterNode(client, settings)


@pytest.fixture(autouse=True)
def patch_trace():
    """Patch trace to avoid tracing in tests."""
    with patch("app.graph.nodes.router.trace") as mock_trace:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = mock_ctx
        yield mock_trace


class TestRouterSimpleQuery:
    """Tests for simple query routing."""

    @pytest.mark.asyncio
    async def test_simple_query_returns_final_answer(self):
        """Simple query with answer should return final_answer and end."""
        router = _make_router({
            "message": {
                "content": '{"is_simple": true, "answer": "Hello, world!"}'
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "hello"}])

        result = await router.action(state)

        assert result["final_answer"] == "Hello, world!"
        assert result["next_action"] == "end"

    @pytest.mark.asyncio
    async def test_simple_query_with_empty_answer(self):
        """Simple query with empty answer should not return final_answer."""
        router = _make_router({
            "message": {
                "content": '{"is_simple": true, "answer": ""}'
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "hello"}])

        result = await router.action(state)

        assert "final_answer" not in result
        assert result["next_action"] == "orchestrator"


class TestRouterComplexQuery:
    """Tests for complex query routing."""

    @pytest.mark.asyncio
    async def test_complex_task_routes_to_reasoning(self):
        """Complex task should route to reasoning."""
        router = _make_router({
            "message": {
                "content": '{"is_complex_task": true}'
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "analyze this"}])

        result = await router.action(state)

        assert result["is_complex_task"] is True
        assert result["next_action"] == "reasoning"

    @pytest.mark.asyncio
    async def test_needs_reasoning_model_routes_to_reasoning(self):
        """Query needing reasoning model should route to reasoning."""
        router = _make_router({
            "message": {
                "content": '{"needs_reasoning_model": true}'
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "explain deeply"}])

        result = await router.action(state)

        assert result["is_complex_task"] is True
        assert result["next_action"] == "reasoning"

    @pytest.mark.asyncio
    async def test_complex_task_takes_precedence_over_simple(self):
        """When both is_simple and is_complex_task are true, complex takes precedence."""
        router = _make_router({
            "message": {
                "content": '{"is_simple": true, "is_complex_task": true}'
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "complex but simple"}])

        result = await router.action(state)

        assert result["is_complex_task"] is True
        assert result["next_action"] == "reasoning"


class TestRouterOrchestratorQuery:
    """Tests for orchestrator query routing."""

    @pytest.mark.asyncio
    async def test_orchestrator_route(self):
        """Non-simple, non-complex query should route to orchestrator."""
        router = _make_router({
            "message": {
                "content": '{"is_simple": false, "is_complex_task": false}'
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "general query"}])

        result = await router.action(state)

        assert result["is_complex_task"] is False
        assert result["next_action"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_fallback_route(self):
        """Invalid JSON response should trigger fallback routing."""
        router = _make_router({
            "message": {
                "content": 'invalid json response'
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "test"}])

        result = await router.action(state)

        assert result["next_action"] == "orchestrator"
        assert "is_complex_task" in result

    @pytest.mark.asyncio
    async def test_empty_response_uses_fallback(self):
        """Empty LLM response should use fallback routing."""
        router = _make_router({
            "message": {
                "content": ''
            }
        })
        state = _make_state(messages=[{"role": "user", "content": "test"}])

        result = await router.action(state)

        assert result["next_action"] == "orchestrator"