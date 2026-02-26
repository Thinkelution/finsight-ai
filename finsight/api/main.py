"""FastAPI application entry point for FinSight AI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finsight.api.routes import alerts, health, market, query
from finsight.config.logging import setup_logging

setup_logging()

app = FastAPI(
    title="FinSight AI",
    description="Real-time financial intelligence powered by RAG + fine-tuned LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, tags=["Query"])
app.include_router(market.router, tags=["Market Data"])
app.include_router(health.router, tags=["Health"])
app.include_router(alerts.router, tags=["Alerts"])


@app.on_event("startup")
async def startup():
    from finsight.storage.qdrant_store import ensure_collection
    try:
        ensure_collection()
    except Exception:
        pass  # non-fatal on startup — Qdrant may not be running yet
