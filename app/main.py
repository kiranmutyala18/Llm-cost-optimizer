from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.embeddings import build_embedder
from app.llm_client import build_llm_client
from app.rate_limiter import RateLimiter
from app.routers import analytics, clients, proxy
from app.semantic_cache import SemanticCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 1. Tables
    init_db()

    # 2. Redis connection (shared for exact-cache + rate limiting)
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    # 3. Embedder + semantic cache, warm the FAISS index from Postgres
    embedder = build_embedder(settings)
    semantic_cache = SemanticCache(
        embedder=embedder,
        redis_client=redis_client,
        similarity_threshold=settings.similarity_threshold,
        exact_ttl_seconds=settings.exact_cache_ttl_seconds,
    )
    db = SessionLocal()
    try:
        loaded = semantic_cache.load_from_db(db)
    finally:
        db.close()

    app.state.redis = redis_client
    app.state.semantic_cache = semantic_cache
    app.state.rate_limiter = RateLimiter(redis_client)
    app.state.llm_client = build_llm_client(settings)
    app.state.cache_entries_loaded = loaded

    yield
    # No explicit teardown needed; connections close with process exit.


app = FastAPI(
    title="LLM Cost & Usage Optimizer",
    description=(
        "Middleware proxy that sits in front of an LLM API, caches semantically "
        "similar prompts to cut redundant calls, enforces per-client rate limits, "
        "and tracks token usage / cost for an analytics dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(clients.router)
app.include_router(proxy.router)
app.include_router(analytics.router)


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "cache_entries_loaded": getattr(app.state, "cache_entries_loaded", 0),
    }
