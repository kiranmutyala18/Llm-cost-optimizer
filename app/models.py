import secrets
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_api_key() -> str:
    return f"sk-llmopt-{secrets.token_urlsafe(32)}"


class Client(Base):
    """A tenant / API consumer of the proxy. Each client has its own key,
    rate limit, and usage history."""

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key: Mapped[str] = mapped_column(String(64), unique=True, index=True, default=_new_api_key)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=60)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    request_logs: Mapped[list["RequestLog"]] = relationship(back_populates="client")


class CacheEntry(Base):
    """A cached prompt/response pair with its embedding, used for both
    exact-match and semantic-similarity lookups. The embedding is persisted
    here so the in-memory FAISS index can be rebuilt on restart."""

    __tablename__ = "cache_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list] = mapped_column(JSON, nullable=False)  # list[float]
    model: Mapped[str] = mapped_column(String(100), default="unknown")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_hit_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RequestLog(Base):
    """One row per inbound proxy request, used to power analytics
    (token usage, cost, cache hit rate, latency) per client."""

    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), nullable=False)
    model: Mapped[str] = mapped_column(String(100), default="unknown")
    cache_status: Mapped[str] = mapped_column(String(20), default="miss")  # exact_hit | semantic_hit | miss
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    client: Mapped["Client"] = relationship(back_populates="request_logs")
