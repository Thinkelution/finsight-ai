"""FastAPI application entry point for FinSight AI."""

import os

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from finsight.api.routes import alerts, feed, health, market, predictions, query
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


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response: Response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(query.router, tags=["Query"])
app.include_router(market.router, tags=["Market Data"])
app.include_router(health.router, tags=["Health"])
app.include_router(alerts.router, tags=["Alerts"])
app.include_router(feed.router, tags=["Data Feed"])
app.include_router(predictions.router, tags=["Predictions"])

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return {"message": "FinSight AI API is running. Visit /docs for API documentation."}


@app.on_event("startup")
async def startup():
    from finsight.storage.qdrant_store import ensure_collection
    try:
        ensure_collection()
    except Exception:
        pass
