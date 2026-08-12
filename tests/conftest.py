"""Tests configuration and shared fixtures."""
import sys
from unittest.mock import MagicMock, Mock

# Patch missing C-extension / DB dependencies so orchestrator tests can import.
def _make_fake_module(name: str):
    m = MagicMock(name=name)
    # Make subpackage imports work (e.g. sqlalchemy.dialects.postgresql)
    type(m).__getitem__ = lambda self, key: m
    return m

fake_pgvector = _make_fake_module("pgvector")
fake_sqlalchemy = _make_fake_module("sqlalchemy")
fake_dialects = _make_fake_module("sqlalchemy.dialects")
fake_postgresql = _make_fake_module("sqlalchemy.dialects.postgresql")
fake_orm = _make_fake_module("sqlalchemy.orm")

sys.modules["pgvector"] = fake_pgvector
sys.modules["pgvector.sqlalchemy"] = fake_pgvector.sqlalchemy = fake_pgvector
sys.modules["sqlalchemy"] = fake_sqlalchemy
sys.modules["sqlalchemy.orm"] = fake_orm
sys.modules["sqlalchemy.ext"] = _make_fake_module("sqlalchemy.ext")
sys.modules["sqlalchemy.ext.declarative"] = fake_sqlalchemy.ext.declarative = fake_sqlalchemy.ext
sys.modules["sqlalchemy.dialects"] = fake_dialects
sys.modules["sqlalchemy.dialects.postgresql"] = fake_postgresql


from unittest.mock import AsyncMock, MagicMock  # noqa: E402, F401

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
