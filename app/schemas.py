from datetime import datetime

from pydantic import BaseModel, Field


# ---------- Clients ----------

class ClientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=100_000)


class ClientOut(BaseModel):
    id: int
    name: str
    api_key: str
    rate_limit_per_minute: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- Chat proxy (OpenAI-style) ----------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list[ChatMessage]
    temperature: float = 0.7


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    content: str
    usage: ChatCompletionUsage
    cache_status: str  # "exact_hit" | "semantic_hit" | "miss"
    similarity_score: float | None = None
    latency_ms: float


# ---------- Analytics ----------

class UsageSummary(BaseModel):
    client_id: int
    client_name: str
    total_requests: int
    exact_hits: int
    semantic_hits: int
    misses: int
    cache_hit_rate: float
    total_tokens: int
    total_cost_usd: float
    estimated_cost_saved_usd: float
    avg_latency_ms: float
