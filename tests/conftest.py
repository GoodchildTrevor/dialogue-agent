"""Tests configuration and shared fixtures."""
from unittest.mock import MagicMock

import pytest

from app.graph.state import AssistantState


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