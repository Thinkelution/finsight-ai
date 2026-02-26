from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Infrastructure
    redis_url: str = "redis://localhost:6379"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "finsight_chunks"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_llm_model: str = "finsight-qwen14b"
    ollama_embed_model: str = "nomic-embed-text"

    # Market Data APIs
    alpha_vantage_api_key: str = ""
    polygon_api_key: str = ""
    news_api_key: str = ""

    # Optional premium APIs
    benzinga_api_key: str = ""
    oanda_api_key: str = ""
    oanda_account_id: str = ""

    # RunPod
    runpod_api_key: str = ""

    # Application
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    alert_price_move_threshold: float = 0.015
    summary_refresh_interval: int = 1800
    news_expiry_days: int = 7

    # Groq fallback
    groq_api_key: str = ""

    # Embedding config
    embed_dim: int = Field(default=768, description="nomic-embed-text dimension")
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 8

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
