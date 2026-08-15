"""Integration tests for app.core.llm_client.LLMClient.

Tests the actual HTTP request/response flow using httpx MockTransport
to simulate LiteLLM-compatible API responses without hitting a real server.
"""

import json
import os
from unittest.mock import patch

import pytest
import httpx

from app.core.config import Settings
from app.core.llm_client import LLMClient


# Minimal env vars needed so Settings() doesn't raise on required fields.
_MINIMAL_ENV = {
    "API_KEY": "test-key",
    "QDRANT_INGESTER_API": "ingest-key",
    "UPLOAD_STORAGE_DIR": "/tmp/uploads",
    "LLM_BASE_URL": "http://127.0.0.1:4736/v1",
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


def _make_client(**settings_overrides) -> LLMClient:
    """Create an LLMClient wired to mock settings."""
    settings = _make_settings(**settings_overrides)
    with patch("app.core.llm_client.get_settings", return_value=settings):
        # Re-create the module-level get_settings so LLMClient picks it up
        import app.core.llm_client as mod
        old_get = mod.get_settings
        mod.get_settings = lambda: settings
        try:
            return LLMClient()
        finally:
            mod.get_settings = old_get


def _transport(status_code=200, body=None, stream_chunks=None):
    """Build an httpx MockTransport from a response."""
    if body is None:
        body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        accept_header = request.headers.get("accept", "")
        is_stream = "text/event-stream" in accept_header or (
            request.url.params.get("stream") == "true"
        )

        if is_stream and stream_chunks:
            events = []
            for chunk in stream_chunks:
                data = json.dumps(chunk)
                events.append(f"data: {data}\n\n".encode())
            events.append(b"data: [DONE]\n\n")
            body_bytes = b"".join(events)
            return httpx.Response(
                status_code,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body_bytes),
            )

        if is_stream and not stream_chunks:
            return httpx.Response(
                status_code,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(b"data: [DONE]\n\n"),
            )

        body_bytes = json.dumps(body).encode()
        return httpx.Response(status_code, content=body_bytes)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# TestChatSuccess — happy path for non-streaming chat
# ---------------------------------------------------------------------------

class TestChatSuccess:
    """Successful non-streaming chat completions."""

    def test_basic_completion(self):
        client = _make_client()
        resp = client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert resp["choices"][0]["message"]["content"] == "ok"

    def test_streaming_mode_returns_chunks(self):
        client = _make_client()
        chunks = list(client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        ))
        assert len(chunks) > 0
        assert all("choices" in c for c in chunks)

    def test_system_prompt_included(self):
        client = _make_client()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        resp = client.chat(messages=messages, stream=False)
        assert resp["choices"][0]["message"]["content"] == "ok"

    def test_temperature_override(self):
        client = _make_client()
        client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )

    def test_custom_headers_passed(self):
        client = _make_client()
        resp = client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            extra_headers={"X-Custom": "value"},
            stream=False,
        )
        assert resp["choices"][0]["message"]["content"] == "ok"

    def test_max_tokens_override(self):
        client = _make_client()
        resp = client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=512,
            stream=False,
        )
        assert resp["choices"][0]["message"]["content"] == "ok"

    def test_multiple_messages(self):
        client = _make_client()
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        resp = client.chat(messages=messages, stream=False)
        assert resp["choices"][0]["message"]["content"] == "ok"


# ---------------------------------------------------------------------------
# TestChatStreaming — streaming mode edge cases
# ---------------------------------------------------------------------------

class TestChatStreaming:
    """Streaming chat completions and error handling."""

    def test_stream_yields_chunks(self):
        client = _make_client()
        chunks = list(client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ))
        assert len(chunks) > 0
        assert isinstance(chunks[0], dict)

    def test_stream_empty_response(self):
        client = _make_client()
        chunks = list(client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ))
        assert isinstance(chunks, list)

    def test_stream_http_error_401(self):
        client = _make_client()
        t = _transport(status_code=401, body={"error": {"message": "Unauthorized"}})
        with pytest.raises(httpx.HTTPStatusError):
            list(client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                transport=t,
            ))

    def test_stream_http_error_500(self):
        client = _make_client()
        t = _transport(status_code=500, body={"error": {"message": "Internal error"}})
        with pytest.raises(httpx.HTTPStatusError):
            list(client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                transport=t,
            ))

    def test_stream_connection_timeout(self):
        client = _make_client()
        t = _transport(status_code=200)
        with patch("httpx.Client.__enter__") as mock_enter:
            mock_enter.side_effect = httpx.ConnectTimeout("timed out")
            with pytest.raises(httpx.ConnectTimeout):
                list(client.chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    stream=True,
                    transport=t,
                ))

    def test_stream_json_parse_error(self):
        client = _make_client()
        bad_chunks = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"invalid json bracket": "[broken"},
        ]
        t = _transport(status_code=200, stream_chunks=bad_chunks)
        chunks = list(client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            transport=t,
        ))
        assert isinstance(chunks, list)

    def test_stream_exception_handling(self):
        client = _make_client()
        t = _transport(status_code=200)
        with patch("httpx.Client.__enter__") as mock_enter:
            mock_enter.side_effect = Exception("something broke")
            with pytest.raises(Exception, match="something broke"):
                list(client.chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    stream=True,
                    transport=t,
                ))


# ---------------------------------------------------------------------------
# TestChatStreamNonStreaming — chat_stream() in non-streaming mode
# ---------------------------------------------------------------------------

class TestChatStreamNonStreaming:
    """chat_stream() called with stream=False returns a list."""

    def test_returns_list(self):
        client = _make_client()
        result = client.chat_stream(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_raises_on_401(self):
        client = _make_client()
        t = _transport(status_code=401)
        with pytest.raises(httpx.HTTPStatusError):
            client.chat_stream(
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                transport=t,
            )

    def test_raises_on_500(self):
        client = _make_client()
        t = _transport(status_code=500)
        with pytest.raises(httpx.HTTPStatusError):
            client.chat_stream(
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                transport=t,
            )


# ---------------------------------------------------------------------------
# TestChatNonStreamingEdgeCases — non-streaming edge cases
# ---------------------------------------------------------------------------

class TestChatNonStreamingEdgeCases:
    """Edge cases for non-streaming chat responses."""

    def test_empty_response_body(self):
        client = _make_client()
        t = _transport(status_code=200, body={})
        resp = client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            transport=t,
        )
        assert isinstance(resp, dict)

    def test_non_json_response(self):
        client = _make_client()
        def handler(request):
            return httpx.Response(200, content=b"not json at all")
        t = httpx.MockTransport(handler)
        resp = client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            transport=t,
        )
        assert isinstance(resp, dict)

    def test_missing_message_field(self):
        client = _make_client()
        t = _transport(body={"choices": [{"message": {}}]})
        resp = client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            transport=t,
        )
        assert isinstance(resp, dict)

    def test_none_content(self):
        client = _make_client()
        t = _transport(body={"choices": [{"message": {"role": "assistant", "content": None}}]})
        resp = client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            transport=t,
        )
        assert isinstance(resp, dict)


# ---------------------------------------------------------------------------
# TestChatStreamNonStreamingEdgeCases — chat_stream non-streaming edge cases
# ---------------------------------------------------------------------------

class TestChatStreamNonStreamingEdgeCases:
    """Edge cases for chat_stream() in non-streaming mode."""

    def test_empty_list_return(self):
        client = _make_client()
        t = _transport(body={"choices": []})
        result = client.chat_stream(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            transport=t,
        )
        assert isinstance(result, list)

    def test_http_error_in_non_streaming(self):
        client = _make_client()
        t = _transport(status_code=403, body={"error": {"message": "Forbidden"}})
        with pytest.raises(httpx.HTTPStatusError):
            client.chat_stream(
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                transport=t,
            )
