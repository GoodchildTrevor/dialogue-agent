from functools import lru_cache

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
    LOG_LEVEL: str = "INFO"

    LLM_BASE_URL: str
    ROUTER_MODEL: str
    REASONING_MODEL: str

    MCP_SERVER_URL: str

    POSTGRES_URL: str

    CHUNKER_SERVICE_URL: str
    PG_INGESTER_URL: str

    MAX_TOOL_RETRIES: int = 3
    HTTP_TIMEOUT_SECONDS: float = 20.0
    TOOL_REQUEST_TIMEOUT_SECONDS: float = 45.0
    HTTP_MAX_CONNECTIONS: int = 100
    HISTORY_SEARCH_LIMIT: int = 5
    API_KEY: str = "default-secret-key-change-me"


def get_settings() -> Settings:
    return Settings()
