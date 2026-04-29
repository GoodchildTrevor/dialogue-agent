<<<<<<< HEAD
"""Tests configuration and shared fixtures."""
from unittest.mock import MagicMock
=======
"""Shared fixtures for routing-logic tests.

All objects are pure in-memory stubs — no network, no DB, no Ollama required.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock
>>>>>>> f6d283305a47cf8b1710c7cd1e4d0b6294657e47

import pytest

from app.graph.state import AssistantState


<<<<<<< HEAD
@pytest.fixture
def base_state() -> AssistantState:
    """Base state for router node tests."""
    return {
        "request_id": "test-request-id",
        "user_id": "test-user-id",
        "messages": [
            {"role": "user", "content": "test message"}
        ],
        "is_complex_task": False,
    }


@pytest.fixture
def make_llm_client():
    """Factory fixture to create LLM client mocks."""
    def _make(value: dict):
        client = MagicMock()
        client.chat = AsyncMock(return_value=value)
        return client
    return _make


@pytest.fixture
def _Settings():
    """Mock settings for router node tests."""
    settings = MagicMock()
    settings.ROUTER_MODEL = "test-model"
    settings.REASONING_MODEL = "test-reasoning-model"
    settings.MAX_TOKENS = 1000
    return settings


class AsyncMock(MagicMock):
    """Mock for async functions."""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
=======
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
>>>>>>> f6d283305a47cf8b1710c7cd1e4d0b6294657e47
