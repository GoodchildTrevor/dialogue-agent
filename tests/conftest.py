"""Shared fixtures for routing-logic tests.

All objects are pure in-memory stubs — no network, no DB, no Ollama required.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.graph.state import AssistantState


# ---------------------------------------------------------------------------
# Minimal settings stub
# ---------------------------------------------------------------------------

class _Settings:
    ROUTER_MODEL: str = "stub-model"


@pytest.fixture()
def settings() -> _Settings:
    return _Settings()


# ---------------------------------------------------------------------------
# LLM client stub factory
# ---------------------------------------------------------------------------

def make_llm_client(content: str) -> AsyncMock:
    """Return an AsyncMock llm_client whose .chat() resolves to *content*."""
    client = AsyncMock()
    client.chat.return_value = {
        "message": {"content": content},
        "prompt_eval_count": 10,
        "eval_count": 20,
    }
    return client


# ---------------------------------------------------------------------------
# AssistantState builders
# ---------------------------------------------------------------------------

def base_state(**overrides: Any) -> AssistantState:
    """Minimal valid AssistantState for router tests."""
    state: AssistantState = {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "u-test",
        "request_id": "r-test",
        "context": {},
        "intermediate_steps": [],
        "is_complex_task": False,
        "next_action": "",
        "final_answer": "",
        "pending_tool_calls": [],
        "tool_results": [],
        "last_tool_error": None,
        "tool_retry_count": 0,
        "status_queue": None,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state
