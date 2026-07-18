import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.deps import get_current_client, get_llm_client, get_rate_limiter, get_semantic_cache
from app.database import get_db
from app.llm_client import BaseLLMClient, estimate_cost_usd
from app.models import Client, RequestLog
from app.rate_limiter import RateLimiter
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage
from app.semantic_cache import SemanticCache

router = APIRouter(prefix="/v1", tags=["proxy"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    payload: ChatCompletionRequest,
    client: Client = Depends(get_current_client),
    db: Session = Depends(get_db),
    cache: SemanticCache = Depends(get_semantic_cache),
    limiter: RateLimiter = Depends(get_rate_limiter),
    llm: BaseLLMClient = Depends(get_llm_client),
):
    start = time.perf_counter()

    # 1. Rate limit check
    rl = limiter.check(client.id, client.rate_limit_per_minute)
    if not rl.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({rl.limit}/min). Retry in {rl.reset_seconds}s.",
        )

    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    # Concatenate the conversation into one cache key text so multi-turn
    # context is part of what we match on.
    prompt_text = "\n".join(f"{m.role}: {m.content}" for m in payload.messages)

    # 2. Cache lookup (exact -> semantic)
    result = cache.lookup(db, prompt_text)

    if result.hit:
        latency_ms = (time.perf_counter() - start) * 1000
        _log_request(
            db, client, payload.model, result.status, result.similarity_score,
            result.prompt_tokens, result.completion_tokens, latency_ms,
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            model=payload.model,
            content=result.response_text,
            usage=ChatCompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            cache_status=result.status,
            similarity_score=result.similarity_score,
            latency_ms=latency_ms,
        )

    # 3. Cache miss -> call the real LLM
    llm_result = llm.complete(
        model=payload.model,
        messages=[m.model_dump() for m in payload.messages],
        temperature=payload.temperature,
    )

    # 4. Write-through cache
    cache.store(
        db,
        raw_prompt=prompt_text,
        response_text=llm_result.text,
        model=payload.model,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    _log_request(
        db, client, payload.model, "miss", None,
        llm_result.prompt_tokens, llm_result.completion_tokens, latency_ms,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        model=payload.model,
        content=llm_result.text,
        usage=ChatCompletionUsage(
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.prompt_tokens + llm_result.completion_tokens,
        ),
        cache_status="miss",
        similarity_score=None,
        latency_ms=latency_ms,
    )


def _log_request(
    db: Session,
    client: Client,
    model: str,
    cache_status: str,
    similarity_score: float | None,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> None:
    cost = 0.0 if cache_status != "miss" else estimate_cost_usd(model, prompt_tokens, completion_tokens)
    log = RequestLog(
        client_id=client.id,
        model=model,
        cache_status=cache_status,
        similarity_score=similarity_score,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=cost,
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()
