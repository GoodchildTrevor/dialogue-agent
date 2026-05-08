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

    LLM_BASE_URL: str
    ROUTER_MODEL: str
    REASONING_MODEL: str

    MCP_SERVER_URL: str
    MCP_AUTH_TOKEN: str

    FILE_CONVERTER_MCP_URL: str = ""
    FILE_CONVERTER_AUTH_TOKEN: str = ""

    POSTGRES_URL: str

    CHUNKER_SERVICE_URL: str
    DISTANCE_THRESHOLD: float = 0.45

    # File ingestion chunking params (forwarded to qdrant-ingester)
    CHUNK_SIZE: int = 512
    OVERLAP: int = 50

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


def get_settings() -> Settings:
    return Settings()
