"""Pydantic request/response models for the FinSight API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    asset_class: str | None = Field(
        default=None,
        description="Filter by asset class: equities, forex, commodities, macro, crypto",
    )
    hours_back: int = Field(default=24, ge=1, le=168)
    stream: bool = Field(default=False, description="Enable streaming response")
    session_id: str | None = Field(
        default=None,
        description="Session ID for multi-turn conversation history",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    live_prices_at: str
    chunks_used: int
    provider: str = "ollama"
    session_id: str | None = None


class MarketPricesResponse(BaseModel):
    rates: dict[str, float]
    changes: dict[str, float]
    timestamp: str
    symbols_fetched: int
    symbols_failed: int


class HealthResponse(BaseModel):
    status: str
    qdrant_status: str | None = None
    redis_status: str | None = None
    ollama_status: str | None = None
    chunks_count: int | None = None
