"""Tests for app.core.config.Settings."""

import json
import os
from unittest.mock import patch

import pytest

from app.core.config import Settings


# Minimal env vars needed so Settings() doesn't raise on required fields.
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


class TestSettingsDefaults:
    """Test default values for Settings fields that have defaults."""

    def test_app_name(self):
        settings = _make_settings()
        assert settings.APP_NAME == "dialogue-bot"

    def test_api_v1_prefix(self):
        settings = _make_settings()
        assert settings.API_V1_PREFIX == "/api/v1"

    def test_log_level(self):
        settings = _make_settings()
        assert settings.LOG_LEVEL == "INFO"

    def test_qdrant_defaults(self):
        settings = _make_settings()
        assert settings.QDRANT_URL == "http://qdrant:6333"
        assert settings.QDRANT_COLLECTION_DOCS == "documents"

    def test_max_upload_size(self):
        settings = _make_settings()
        assert settings.MAX_UPLOAD_SIZE_MB == 50

    def test_allowed_mime_types_default(self):
        settings = _make_settings()
        assert len(settings.ALLOWED_MIME_TYPES) == 3
        assert "application/pdf" in settings.ALLOWED_MIME_TYPES

    def test_litellm_master_key_empty_by_default(self):
        settings = _make_settings()
        assert settings.LITELLM_MASTER_KEY == ""

    def test_file_converter_mcp_empty_by_default(self):
        settings = _make_settings()
        assert settings.FILE_CONVERTER_MCP_URL == ""
        assert settings.FILE_CONVERTER_AUTH_TOKEN == ""

    def test_file_auto_attach_minutes(self):
        settings = _make_settings()
        assert settings.FILE_AUTO_ATTACH_MINUTES == 30

    def test_distance_threshold(self):
        settings = _make_settings()
        assert settings.DISTANCE_THRESHOLD == 0.45

    def test_chunking_defaults(self):
        settings = _make_settings()
        assert settings.CHUNK_SIZE == 512
        assert settings.OVERLAP == 50

    def test_inline_threshold_none_by_default(self):
        settings = _make_settings()
        assert settings.INLINE_THRESHOLD is None

    def test_max_tool_retries(self):
        settings = _make_settings()
        assert settings.MAX_TOOL_RETRIES == 3

    def test_http_timeout(self):
        settings = _make_settings()
        assert settings.HTTP_TIMEOUT_SECONDS == 20.0

    def test_orchestrator_timeout(self):
        settings = _make_settings()
        assert settings.ORCHESTRATOR_TIMEOUT_SECONDS == 120.0

    def test_reasoning_timeout(self):
        settings = _make_settings()
        assert settings.REASONING_TIMEOUT_SECONDS == 300.0

    def test_http_max_connections(self):
        settings = _make_settings()
        assert settings.HTTP_MAX_CONNECTIONS == 100

    def test_history_search_limit(self):
        settings = _make_settings()
        assert settings.HISTORY_SEARCH_LIMIT == 5

    def test_summarization_model_empty_by_default(self):
        settings = _make_settings()
        assert settings.SUMMARIZATION_MODEL == ""

    def test_summarization_max_input_chars(self):
        settings = _make_settings()
        assert settings.SUMMARIZATION_MAX_INPUT_CHARS == 64000


class TestMCPServersList:
    """Test the mcp_servers_list property."""

    def test_primary_server_always_present(self):
        settings = _make_settings()
        servers = settings.mcp_servers_list
        assert len(servers) >= 1
        primary = servers[0]
        assert primary["name"] == "primary"
        assert primary["url"] == "http://mcp:3000/mcp"
        assert primary["token"] == "mcp-token"

    def test_mcp_servers_json_parsed(self):
        extra = json.dumps([{"url": "http://oracle:3000/mcp", "token": "oracle-token", "name": "oracle"}])
        settings = _make_settings(MCP_SERVERS=extra)
        servers = settings.mcp_servers_list
        assert len(servers) == 2
        assert servers[1]["name"] == "oracle"
        assert servers[1]["url"] == "http://oracle:3000/mcp"
        assert servers[1]["token"] == "oracle-token"

    def test_mcp_servers_defaults_token_and_name(self):
        extra = json.dumps([{"url": "http://fallback:3000/mcp"}])
        settings = _make_settings(MCP_SERVERS=extra)
        servers = settings.mcp_servers_list
        assert servers[1]["token"] == ""
        assert servers[1]["name"] == "http://fallback:3000/mcp"

    def test_file_converter_mcp_appended_when_set(self):
        settings = _make_settings(
            FILE_CONVERTER_MCP_URL="http://converter:8000/mcp",
            FILE_CONVERTER_AUTH_TOKEN="conv-token",
        )
        servers = settings.mcp_servers_list
        assert len(servers) == 2  # primary + file-converter
        assert servers[1]["name"] == "file-converter"
        assert servers[1]["url"] == "http://converter:8000/mcp"
        assert servers[1]["token"] == "conv-token"

    def test_file_converter_mcp_not_appended_when_empty(self):
        settings = _make_settings(FILE_CONVERTER_MCP_URL="")
        servers = settings.mcp_servers_list
        assert len(servers) == 1

    def test_all_three_server_types(self):
        extra = json.dumps([{"url": "http://oracle:3000/mcp", "name": "oracle"}])
        settings = _make_settings(
            MCP_SERVERS=extra,
            FILE_CONVERTER_MCP_URL="http://converter:8000/mcp",
        )
        servers = settings.mcp_servers_list
        assert len(servers) == 3
        assert servers[0]["name"] == "primary"
        assert servers[1]["name"] == "oracle"
        assert servers[2]["name"] == "file-converter"

    def test_multiple_extra_servers(self):
        extra = json.dumps([
            {"url": "http://a:3000/mcp", "name": "server-a"},
            {"url": "http://b:3000/mcp", "name": "server-b"},
        ])
        settings = _make_settings(MCP_SERVERS=extra)
        servers = settings.mcp_servers_list
        assert len(servers) == 3  # primary + a + b


class TestEnvironmentVariableOverrides:
    """Test that environment variables override defaults."""

    def test_log_level_override(self):
        settings = _make_settings(LOG_LEVEL="DEBUG")
        assert settings.LOG_LEVEL == "DEBUG"

    def test_qdrant_url_override(self):
        settings = _make_settings(QDRANT_URL="http://custom-qdrant:6333")
        assert settings.QDRANT_URL == "http://custom-qdrant:6333"

    def test_max_upload_size_override(self):
        settings = _make_settings(MAX_UPLOAD_SIZE_MB="100")
        assert settings.MAX_UPLOAD_SIZE_MB == 100

    def test_distance_threshold_override(self):
        settings = _make_settings(DISTANCE_THRESHOLD="0.75")
        assert settings.DISTANCE_THRESHOLD == 0.75

    def test_chunk_size_override(self):
        settings = _make_settings(CHUNK_SIZE="256")
        assert settings.CHUNK_SIZE == 256

    def test_inline_threshold_enabled(self):
        settings = _make_settings(INLINE_THRESHOLD="100")
        assert settings.INLINE_THRESHOLD == 100

    def test_summarization_model_enabled(self):
        settings = _make_settings(SUMMARIZATION_MODEL="qwen2.5-0.5b-instruct")
        assert settings.SUMMARIZATION_MODEL == "qwen2.5-0.5b-instruct"


class TestSettingsRequiredFields:
    """Test that required fields are enforced."""

    def test_required_fields_present(self):
        settings = _make_settings()
        # All required fields should be set (not raise)
        assert settings.API_KEY == "test-key"
        assert settings.QDRANT_INGESTER_API == "ingest-key"
        assert settings.UPLOAD_STORAGE_DIR == "/tmp/uploads"
        assert settings.LLM_BASE_URL == "https://api.example.com/v1"
        assert settings.ROUTER_MODEL == "router-model"
        assert settings.REASONING_MODEL == "reasoning-model"
        assert settings.MCP_SERVER_URL == "http://mcp:3000/mcp"
        assert settings.MCP_AUTH_TOKEN == "mcp-token"
        assert settings.POSTGRES_URL == "postgresql://localhost/test"
        assert settings.CHUNKER_SERVICE_URL == "http://chunker:8000"
        assert settings.EMBEDDING_API_URL == "http://embedding:8000"
        assert settings.EMBEDDING_MODEL_NAME == "bge-large"
        assert settings.EMBEDDING_BATCH_SIZE == 32
        assert settings.EMBEDDING_INSERT_BATCH_SIZE == 16


class TestSettingsTypeCoercion:
    """Test that string env vars are coerced to correct types."""

    def test_max_upload_size_int(self):
        settings = _make_settings(MAX_UPLOAD_SIZE_MB="75")
        assert isinstance(settings.MAX_UPLOAD_SIZE_MB, int)
        assert settings.MAX_UPLOAD_SIZE_MB == 75

    def test_distance_threshold_float(self):
        settings = _make_settings(DISTANCE_THRESHOLD="0.6")
        assert isinstance(settings.DISTANCE_THRESHOLD, float)
        assert settings.DISTANCE_THRESHOLD == 0.6

    def test_chunk_size_int(self):
        settings = _make_settings(CHUNK_SIZE="1024")
        assert isinstance(settings.CHUNK_SIZE, int)
        assert settings.CHUNK_SIZE == 1024

    def test_inline_threshold_int(self):
        settings = _make_settings(INLINE_THRESHOLD="500")
        assert isinstance(settings.INLINE_THRESHOLD, int)
        assert settings.INLINE_THRESHOLD == 500

    def test_embedding_batch_size_int(self):
        settings = _make_settings(EMBEDDING_BATCH_SIZE="64", EMBEDDING_INSERT_BATCH_SIZE="32")
        assert isinstance(settings.EMBEDDING_BATCH_SIZE, int)
        assert isinstance(settings.EMBEDDING_INSERT_BATCH_SIZE, int)


class TestMCPServersListEdgeCases:
    """Edge cases for mcp_servers_list."""

    def test_mcp_servers_invalid_json_raises_on_access(self):
        settings = _make_settings(MCP_SERVERS="not-valid-json")
        with pytest.raises(json.JSONDecodeError):
            _ = settings.mcp_servers_list

    def test_mcp_servers_empty_array(self):
        settings = _make_settings(MCP_SERVERS="[]")
        servers = settings.mcp_servers_list
        assert len(servers) == 1  # only primary
        assert servers[0]["name"] == "primary"


class TestGetSettings:
    """Test the get_settings() cached function."""

    def test_returns_settings_instance(self):
        from app.core.config import get_settings
        with patch.dict(os.environ, _MINIMAL_ENV, clear=False):
            # Clear cache first
            get_settings.cache_clear()
            settings = get_settings()
        assert isinstance(settings, Settings)
