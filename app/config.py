from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "sqlite:///./llm_optimizer.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Semantic cache
    embedding_provider: str = "local"  # "local" | "openai"
    embedding_dim: int = 256
    similarity_threshold: float = 0.92
    exact_cache_ttl_seconds: int = 86400

    # Upstream LLM
    llm_provider: str = "mock"  # "mock" | "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # Rate limiting
    default_rate_limit_per_minute: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
