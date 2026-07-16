from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "dialogue-bot"
    API_V1_PREFIX: str = "/api/v1"
    API_KEY: str
    LOG_LEVEL: str = "INFO"

    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION_DOCS: str = "documents"
    QDRANT_INGESTER_URL: str = "http://qdrant-ingester:8001"
    QDRANT_INGESTER_API: str
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_STORAGE_DIR: str
    ALLOWED_MIME_TYPES: list[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    # When uploaded_files is empty in a chat/stream request, automatically attach
    # the user's recently indexed files (created within this many minutes).
    # Set to 0 to disable auto-attach.
    FILE_AUTO_ATTACH_MINUTES: int = 30

    LLM_BASE_URL: str
    LITELLM_MASTER_KEY: str = ""  # Bearer token for LiteLLM proxy auth
    ROUTER_MODEL: str
    REASONING_MODEL: str

    # Primary MCP server (always loaded)
    MCP_SERVER_URL: str
    MCP_AUTH_TOKEN: str

    # Legacy file-converter MCP (kept for backward compatibility; prefer MCP_SERVERS)
    FILE_CONVERTER_MCP_URL: str = ""
    FILE_CONVERTER_AUTH_TOKEN: str = ""

    # Additional MCP servers as a JSON array:
    # [{"url": "http://oracle-mcp:3000/mcp", "token": "secret", "name": "oracle"}, ...]
    # The primary MCP_SERVER_URL is always prepended automatically.
    MCP_SERVERS: str = "[]"

    POSTGRES_URL: str

    CHUNKER_SERVICE_URL: str
    DISTANCE_THRESHOLD: float = 0.45

    # File ingestion chunking params (forwarded to qdrant-ingester)
    CHUNK_SIZE: int = 512
    OVERLAP: int = 50

    # If set, documents with token_count <= INLINE_THRESHOLD are returned as
    # raw text instead of being ingested into Qdrant.
    # Set to 0 or leave unset to always ingest.
    INLINE_THRESHOLD: int | None = None

    # Embedding (in-process via fastembed)
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_BATCH_SIZE: int
    EMBEDDING_INSERT_BATCH_SIZE: int

    MAX_TOOL_RETRIES: int = 3
    HTTP_TIMEOUT_SECONDS: float = 20.0
    TOOL_REQUEST_TIMEOUT_SECONDS: float = 45.0
    ORCHESTRATOR_TIMEOUT_SECONDS: float = 120.0
    REASONING_TIMEOUT_SECONDS: float = 300.0
    HTTP_MAX_CONNECTIONS: int = 100
    HISTORY_SEARCH_LIMIT: int = 5

    @property
    def mcp_servers_list(self) -> list[dict[str, Any]]:
        """Return the full ordered list of MCP server configs.

        The primary server (MCP_SERVER_URL / MCP_AUTH_TOKEN) is always
        first. Servers declared in MCP_SERVERS are appended after.
        The legacy FILE_CONVERTER_MCP_URL is appended last when set,
        so existing deployments keep working without changes.

        Each entry is a dict with keys:
            - ``url``   -- full MCP endpoint URL
            - ``token`` -- Bearer auth token (may be empty string)
            - ``name``  -- human-readable label used in logs
        """
        servers: list[dict[str, Any]] = [
            {"url": self.MCP_SERVER_URL, "token": self.MCP_AUTH_TOKEN, "name": "primary"},
        ]

        for entry in json.loads(self.MCP_SERVERS):
            servers.append(
                {
                    "url": entry["url"],
                    "token": entry.get("token", ""),
                    "name": entry.get("name", entry["url"]),
                }
            )

        if self.FILE_CONVERTER_MCP_URL:
            servers.append(
                {
                    "url": self.FILE_CONVERTER_MCP_URL,
                    "token": self.FILE_CONVERTER_AUTH_TOKEN,
                    "name": "file-converter",
                }
            )

        return servers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
