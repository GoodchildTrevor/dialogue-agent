"""Tests for app.graph.tool_registry — ToolRegistry class and _normalize_content helper."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock fastmcp since it's not installed in this environment.
sys.modules["fastmcp"] = MagicMock()
sys.modules["fastmcp"].Client = MagicMock

from app.core.config import Settings
from app.graph.tool_registry import (
    ToolContext,
    ToolRegistry,
    _normalize_content,
)


# ── Minimal env vars so Settings() doesn't raise on required fields ──────────

_MINIMAL_ENV = {
    "API_KEY": "test-key",
    "QDRANT_INGESTER_API": "ingest-key",
    "UPLOAD_STORAGE_DIR": "/tmp/uploads",
    "LLM_BASE_URL": "https://api.example.com/v1",
    "ROUTER_MODEL": "router-model",
    "REASONING_MODEL": "reasoning-model",
    "MCP_SERVER_URL": "http://mcp:3000/mcp",
    "MCP_AUTH_TOKEN": "mcp-token",
    "POSTGRES_URL": "postgresql://localhost/test",
    "CHUNKER_SERVICE_URL": "http://chunker:8000",
    "EMBEDDING_API_URL": "http://embedding:8000",
    "EMBEDDING_MODEL_NAME": "bge-large",
    "EMBEDDING_BATCH_SIZE": "32",
    "EMBEDDING_INSERT_BATCH_SIZE": "16",
}


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance with optional env-var overrides."""
    env = dict(_MINIMAL_ENV, **overrides)
    with patch.dict(os.environ, env, clear=False):
        return Settings()


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def settings():
    return _make_settings()


@pytest.fixture()
def mock_mcp_client():
    # Plain MagicMock (no spec) so __aenter__/__aexit__ can be set.
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.fixture()
def registry(settings, mock_mcp_client):
    return ToolRegistry(settings=settings, mcp_client=mock_mcp_client, name="test-server")


# ── _normalize_content ─────────────────────────────────────────────────────

class TestNormalizeContentString:
    def test_plain_string_returns_as_is(self):
        text, images = _normalize_content("Hello world")
        assert text == "Hello world"
        assert images == []

    def test_string_with_image_regex_extracts_data(self):
        content = 'type="image" data="iVBORw0KGgoAAAANSUhEU"'
        text, images = _normalize_content(content)
        assert "[IMAGE DATA]" in text
        assert len(images) == 1
        assert images[0]["mime_type"] == "image/png"
        assert images[0]["data"] == "iVBORw0KGgoAAAANSUhEU"

    def test_string_with_case_insensitive_image_match(self):
        content = 'TYPE="IMAGE" data="abc123"'
        text, images = _normalize_content(content)
        assert len(images) == 1
        assert images[0]["data"] == "abc123"

    def test_string_with_multiple_image_data_blocks(self):
        content = 'type="image" data="aaa" and type="image" data="bbb"'
        text, images = _normalize_content(content)
        assert len(images) == 2
        assert images[0]["data"] == "aaa"
        assert images[1]["data"] == "bbb"

    def test_string_without_image_pattern_returns_untouched(self):
        content = 'some random text with no image markers'
        text, images = _normalize_content(content)
        assert text == content
        assert images == []


class TestNormalizeContentList:
    def test_list_of_strings_joined_with_newlines(self):
        content = ["hello", "world"]
        text, images = _normalize_content(content)
        assert text == "hello\nworld"
        assert images == []

    def test_list_with_image_dict(self):
        content = [{"type": "image", "mimeType": "image/jpeg", "data": "abc123"}]
        text, images = _normalize_content(content)
        assert len(images) == 1
        assert images[0]["mime_type"] == "image/jpeg"
        assert images[0]["data"] == "abc123"
        assert "[IMAGE:1]" in text

    def test_list_with_text_dict(self):
        content = [{"type": "text", "text": "some result"}]
        text, images = _normalize_content(content)
        assert text == "some result"
        assert images == []

    def test_list_with_mixed_items(self):
        content = [
            "intro",
            {"type": "image", "mimeType": "image/png", "data": "img1"},
            "outro",
        ]
        text, images = _normalize_content(content)
        assert len(images) == 1
        assert images[0]["mime_type"] == "image/png"
        assert images[0]["data"] == "img1"
        lines = text.split("\n")
        assert "intro" in lines
        assert "outro" in lines
        assert "[IMAGE:1]" in text

    def test_list_with_object_having_text_attr(self):
        class Obj:
            text = "object text"

        content = [Obj()]
        text, images = _normalize_content(content)
        assert text == "object text"
        assert images == []

    def test_list_with_non_string_non_dict_non_image_uses_str(self):
        content = [42, None]
        text, images = _normalize_content(content)
        assert text == "42\nNone"
        assert images == []


class TestNormalizeContentDict:
    def test_plain_text_dict(self):
        content = {"type": "text", "text": "hello"}
        text, images = _normalize_content(content)
        assert text == "hello"
        assert images == []

    def test_image_dict(self):
        content = {"type": "image", "mimeType": "image/png", "data": "base64data"}
        text, images = _normalize_content(content)
        assert len(images) == 1
        assert images[0]["mime_type"] == "image/png"
        assert images[0]["data"] == "base64data"

    def test_image_dict_defaults_to_png(self):
        content = {"type": "image", "data": "abc"}
        text, images = _normalize_content(content)
        assert images[0]["mime_type"] == "image/png"

    def test_dict_without_text_key_uses_str(self):
        content = {"foo": "bar"}
        text, images = _normalize_content(content)
        assert text == "{'foo': 'bar'}"


class TestNormalizeContentOther:
    def test_integer(self):
        text, images = _normalize_content(123)
        assert text == "123"
        assert images == []

    def test_none(self):
        text, images = _normalize_content(None)
        assert text == "None"
        assert images == []


# ── ToolRegistry ───────────────────────────────────────────────────────────

class TestToolRegistryHasTool:
    def test_has_tool_returns_false_when_empty(self, registry):
        assert registry.has_tool("any_tool") is False

    def test_has_tool_returns_true_after_refresh(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[
            {"name": "search_files", "description": "Search files"},
        ])
        asyncio.run(registry.refresh_tools())
        assert registry.has_tool("search_files") is True

    def test_has_tool_returns_false_for_missing_tool(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[
            {"name": "only_one", "description": "Only tool"},
        ])
        asyncio.run(registry.refresh_tools())
        assert registry.has_tool("not_a_tool") is False


class TestToolRegistryIsEmpty:
    def test_is_empty_when_no_tools_loaded(self, registry):
        assert registry.is_empty is True

    def test_is_not_empty_after_refresh(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool1", "description": "d"}])
        asyncio.run(registry.refresh_tools())
        assert registry.is_empty is False


class TestToolRegistryStartup:
    def test_startup_calls_refresh_tools(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool1", "description": "d"}])
        asyncio.run(registry.startup())
        assert registry.has_tool("tool1") is True


class TestToolRegistryShutdown:
    def test_shutdown_is_noop(self, registry):
        # Should not raise and does nothing
        asyncio.run(registry.shutdown())


class TestToolRegistryRefreshTools:
    def test_refresh_tools_populates_tools(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[
            {"name": "tool_a", "description": "A tool"},
            {"name": "tool_b", "description": "B tool"},
        ])
        asyncio.run(registry.refresh_tools())
        assert len(registry._tools) == 2
        assert registry.has_tool("tool_a") is True
        assert registry.has_tool("tool_b") is True

    def test_refresh_tools_handles_non_dict_tool_objects(self, registry, mock_mcp_client):
        tool_obj = MagicMock()
        tool_obj.name = "obj_tool"
        tool_obj.description = "Object tool"
        mock_mcp_client.list_tools = AsyncMock(return_value=[tool_obj])
        asyncio.run(registry.refresh_tools())
        assert registry.has_tool("obj_tool") is True

    def test_refresh_tools_skips_tools_without_name(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[
            {"description": "no name"},  # no 'name' key
            {"name": "valid", "description": "ok"},
        ])
        asyncio.run(registry.refresh_tools())
        assert len(registry._tools) == 1

    def test_refresh_tools_skips_tool_object_without_name_attr(self, registry, mock_mcp_client):
        tool_obj = MagicMock()
        del tool_obj.name  # remove name attr
        mock_mcp_client.list_tools = AsyncMock(return_value=[tool_obj])
        asyncio.run(registry.refresh_tools())
        assert registry.is_empty is True

    def test_refresh_tools_clears_cache(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool1", "description": "d"}])
        asyncio.run(registry.refresh_tools())
        cache_before = registry.describe_for_model()
        assert len(cache_before) == 1

        # Refresh again with same tools — should reuse cached tool list (early return)
        mock_mcp_client.list_tools.reset_mock()
        asyncio.run(registry.refresh_tools())
        # list_tools should NOT be called again because _tools is already populated
        mock_mcp_client.list_tools.assert_not_called()

    def test_refresh_tools_handles_exception(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(side_effect=ConnectionError("server down"))
        asyncio.run(registry.refresh_tools())
        # Should not raise — error is logged
        assert registry.is_empty is True


class TestToolRegistryDescribeForModel:
    def test_describe_for_model_returns_list_of_dicts(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[
            {"name": "search", "description": "Search the web", "inputSchema": {"type": "object"}},
        ])
        asyncio.run(registry.refresh_tools())
        descriptions = registry.describe_for_model()
        assert len(descriptions) == 1
        assert descriptions[0]["name"] == "search"
        assert descriptions[0]["description"] == "Search the web"
        assert descriptions[0]["parameters"]["type"] == "object"

    def test_describe_for_model_caches_results(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool1", "description": "d"}])
        asyncio.run(registry.refresh_tools())
        first_call = registry.describe_for_model()
        second_call = registry.describe_for_model()
        assert first_call is second_call  # same cached list object

    def test_describe_for_model_handles_dict_with_parameters_key(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[
            {"name": "tool", "description": "d", "parameters": {"schema": True}},
        ])
        asyncio.run(registry.refresh_tools())
        descriptions = registry.describe_for_model()
        assert descriptions[0]["parameters"] == {"schema": True}

    def test_describe_for_model_handles_object_tool(self, registry, mock_mcp_client):
        tool_obj = MagicMock()
        tool_obj.name = "obj_tool"
        tool_obj.description = "Object description"
        tool_obj.inputSchema = {"type": "object"}
        mock_mcp_client.list_tools = AsyncMock(return_value=[tool_obj])
        asyncio.run(registry.refresh_tools())
        descriptions = registry.describe_for_model()
        assert len(descriptions) == 1
        assert descriptions[0]["name"] == "obj_tool"

    def test_describe_for_model_with_empty_tools(self, registry):
        descriptions = registry.describe_for_model()
        assert descriptions == []


class TestToolRegistryInvoke:
    def test_invoke_known_tool_success(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "search", "description": "d"}])
        asyncio.run(registry.refresh_tools())

        mock_mcp_client.call_tool = AsyncMock(return_value=[{"type": "text", "text": "result"}])

        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("search", {"q": "hello"}, context))

        assert result["ok"] is True
        assert result["tool"] == "search"
        assert result["result"]["content"] == "result"
        assert result["result"]["images"] == []

    def test_invoke_known_tool_with_image_result(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "gen", "description": "d"}])
        asyncio.run(registry.refresh_tools())

        mock_mcp_client.call_tool = AsyncMock(return_value=[
            {"type": "image", "mimeType": "image/png", "data": "base64data"},
            {"type": "text", "text": "done"},
        ])

        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("gen", {}, context))

        assert result["ok"] is True
        assert len(result["result"]["images"]) == 1
        assert result["result"]["images"][0]["mime_type"] == "image/png"
        assert result["result"]["images"][0]["data"] == "base64data"

    def test_invoke_unknown_tool_returns_error(self, registry):
        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("nonexistent", {}, context))

        assert result["ok"] is False
        assert result["tool"] == "nonexistent"
        assert "Unknown tool: nonexistent" in result["error"]["message"]

    def test_invoke_with_emit_status(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "search", "description": "d"}])
        asyncio.run(registry.refresh_tools())
        mock_mcp_client.call_tool = AsyncMock(return_value=[{"type": "text", "text": "ok"}])

        status_events = []

        async def emit_status(msg):
            status_events.append(msg)

        context = ToolContext(user_id="u1", state={}, emit_status=emit_status)
        asyncio.run(registry.invoke("search", {}, context))

        assert len(status_events) == 1
        assert "Calling tool: search" in status_events[0]

    def test_invoke_dict_result_from_call_tool(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool", "description": "d"}])
        asyncio.run(registry.refresh_tools())

        mock_mcp_client.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": "dict content"}],
        })

        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("tool", {}, context))

        assert result["ok"] is True
        assert result["result"]["content"] == "dict content"

    def test_invoke_exception_returns_error(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "fail", "description": "d"}])
        asyncio.run(registry.refresh_tools())

        mock_mcp_client.call_tool = AsyncMock(side_effect=ValueError("something broke"))

        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("fail", {}, context))

        assert result["ok"] is False
        assert result["tool"] == "fail"
        assert result["error"]["message"] == "something broke"
        assert result["error"]["detail"] == "something broke"

    def test_invoke_utf8_error_gives_user_friendly_message(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "binary_tool", "description": "d"}])
        asyncio.run(registry.refresh_tools())

        mock_mcp_client.call_tool = AsyncMock(
            side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid utf-8 sequence")
        )

        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("binary_tool", {}, context))

        assert result["ok"] is False
        assert "non-UTF8/binary data" in result["error"]["message"]
        assert "invalid utf-8 sequence" in result["error"]["detail"]

    def test_invoke_error_serialization_gives_user_friendly_message(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool", "description": "d"}])
        asyncio.run(registry.refresh_tools())

        mock_mcp_client.call_tool = AsyncMock(
            side_effect=Exception("error serializing result to JSON")
        )

        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("tool", {}, context))

        assert "non-UTF8/binary data" in result["error"]["message"]

    def test_invoke_cannot_encode_gives_user_friendly_message(self, registry, mock_mcp_client):
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool", "description": "d"}])
        asyncio.run(registry.refresh_tools())

        mock_mcp_client.call_tool = AsyncMock(
            side_effect=Exception("cannot encode bytes to JSON")
        )

        context = ToolContext(user_id="u1", state={})
        result = asyncio.run(registry.invoke("tool", {}, context))

        assert "non-UTF8/binary data" in result["error"]["message"]


class TestToolRegistryRefreshLock:
    def test_refresh_tools_uses_lock(self, registry, mock_mcp_client):
        """Second refresh should not call list_tools because tools are already loaded."""
        mock_mcp_client.list_tools = AsyncMock(return_value=[{"name": "tool1", "description": "d"}])
        asyncio.run(registry.refresh_tools())
        mock_mcp_client.list_tools.reset_mock()

        # Second call — should short-circuit via `if not self.is_empty: return`
        asyncio.run(registry.refresh_tools())
        mock_mcp_client.list_tools.assert_not_called()


class TestToolContext:
    def test_tool_context_defaults(self):
        ctx = ToolContext(user_id="u1", state={})
        assert ctx.user_id == "u1"
        assert ctx.state == {}
        assert ctx.emit_status is None

    def test_tool_context_with_emit_status(self):
        async def emit(msg):
            pass

        ctx = ToolContext(user_id="u1", state={"key": "val"}, emit_status=emit)
        assert ctx.user_id == "u1"
        assert ctx.state["key"] == "val"
        assert ctx.emit_status is emit
