# LLM Cost & Usage Optimizer

Middleware proxy service that sits in front of an LLM API. It intercepts
requests, caches semantically similar prompts to cut redundant calls,
enforces per-client rate limits, and tracks token usage and cost for an
analytics dashboard.

**Stack:** FastAPI · Redis · FAISS (+ PostgreSQL for durable vector storage) · PostgreSQL

## How the cache works

Every incoming prompt is checked against two tiers before the request is
ever forwarded to the real LLM:

1. **Exact-match tier (Redis).** The prompt is normalized and hashed; an
   O(1) lookup catches byte-identical repeats instantly.
2. **Semantic tier (FAISS + Postgres).** On a miss, the prompt is embedded
   and searched against an in-memory FAISS index of past prompt vectors
   (cosine similarity via normalized inner product). If the nearest neighbor
   is above `SIMILARITY_THRESHOLD` (default `0.92`), the cached response is
   reused — this is what catches *reworded* duplicates ("what's the capital
   of France" vs. "what is the capital of France, please?").

Every new (non-cached) response is written through to Postgres (source of
truth + FAISS-index durability across restarts) and Redis (fast-path cache).

Rate limiting is a Redis fixed-window counter per client, checked before
any cache/LLM work happens.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build
```

The API is now on `http://localhost:8000`. Interactive docs: `/docs`.

## Quick start (local, no Docker)

```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env
# Point DATABASE_URL/REDIS_URL at local instances, or leave LLM_PROVIDER=mock
# and EMBEDDING_PROVIDER=local to run fully offline with no external services.
uvicorn app.main:app --reload
```

## Running the test suite

Tests run fully offline (in-memory SQLite + fakeredis + mock LLM), no
Docker or network required:

```bash
pytest -v
```

## API walkthrough

**1. Create a client (issues an API key):**
```bash
curl -X POST localhost:8000/clients \
  -H "Content-Type: application/json" \
  -d '{"name": "acme-corp", "rate_limit_per_minute": 60}'
```

**2. Send a chat completion through the proxy:**
```bash
curl -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-llmopt-..." \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```
Response includes `cache_status`: `"miss"`, `"exact_hit"`, or `"semantic_hit"`,
plus `similarity_score` for semantic hits.

**3. Check usage analytics:**
```bash
curl localhost:8000/analytics/usage/1
curl localhost:8000/analytics/timeseries/1
```
Returns request counts, cache hit rate, total tokens, estimated spend, and
estimated dollars saved by caching.

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `EMBEDDING_PROVIDER` | `local` (free, offline hashing embedder) or `openai` (production-quality) |
| `SIMILARITY_THRESHOLD` | Cosine similarity cutoff for a semantic cache hit (0–1) |
| `LLM_PROVIDER` | `mock` (offline, deterministic) or `openai` (real upstream calls) |
| `DEFAULT_RATE_LIMIT_PER_MINUTE` | Fallback per-client limit if not set at creation |

Ships defaulted to `local`/`mock` so the whole thing runs with zero API
keys and zero network access — flip both to `openai` and set
`OPENAI_API_KEY` for production use against real embeddings and a real LLM.

## Known limitations / what I'd change for production

Being upfront about the gaps rather than hiding them:

- **`local` embedder is a hashing trick, not real semantics.** It's free and
  offline, and reliably catches near-duplicate phrasing (same words,
  reordered or padded), but it won't catch true paraphrases with different
  vocabulary. The `openai` embedder path exists for that but hasn't been
  load-tested against a live key.
- **FAISS index access isn't thread-safe.** FastAPI runs sync endpoints in a
  threadpool, so concurrent requests hitting `index.add_with_ids()` /
  `index.search()` at the same time is a real race condition. Fine at demo
  traffic, needs a lock (or a move to `IndexIDMap2` behind a mutex, or
  pgvector doing the search instead) before real concurrent load.
- **No auth on `/clients`.** Anyone can mint an API key and read anyone
  else's analytics. Needs an admin-only guard before this is anything more
  than a local demo.
- **Rate limiting is fixed-window, not sliding.** A client can burst up to
  ~2x their limit right at a minute boundary. A sliding-window or
  token-bucket algorithm would close that.
- **Cost figures are estimates from a static pricing table**, not real
  provider billing data — good enough for a relative "cache saved you ~X%"
  signal, not for an invoice.
- **Analytics endpoints load all logs into memory per client.** Fine at demo
  scale; needs to move to SQL-side aggregation (`GROUP BY`, window functions)
  before it's usable at real request volume.

## Project layout

```
app/
  main.py            FastAPI app + startup wiring (Redis, FAISS, LLM client)
  config.py           Settings (env-driven)
  database.py          SQLAlchemy engine/session
  models.py             Client, CacheEntry, RequestLog tables
  schemas.py             Pydantic request/response models
  embeddings.py            Pluggable embedder (local hash / OpenAI)
  llm_client.py             Pluggable upstream LLM (mock / OpenAI) + cost estimation
  semantic_cache.py          Two-tier cache (Redis exact + FAISS semantic)
  rate_limiter.py             Redis fixed-window rate limiter
  deps.py                      Auth + shared dependency wiring
  routers/
    clients.py                  Client/API-key management
    proxy.py                     Core /v1/chat/completions endpoint
    analytics.py                  Usage dashboard endpoints
tests/                             Full offline test suite (SQLite + fakeredis)
docker-compose.yml                  Postgres + Redis + API
```
