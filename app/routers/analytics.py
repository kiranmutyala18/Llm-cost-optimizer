from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.llm_client import estimate_cost_usd
from app.models import Client, RequestLog
from app.schemas import UsageSummary

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _summarize(db: Session, client: Client) -> UsageSummary:
    logs = db.query(RequestLog).filter(RequestLog.client_id == client.id).all()

    total_requests = len(logs)
    exact_hits = sum(1 for l in logs if l.cache_status == "exact_hit")
    semantic_hits = sum(1 for l in logs if l.cache_status == "semantic_hit")
    misses = sum(1 for l in logs if l.cache_status == "miss")
    total_tokens = sum(l.total_tokens for l in logs)
    total_cost = sum(l.estimated_cost_usd for l in logs)
    avg_latency = sum(l.latency_ms for l in logs) / total_requests if total_requests else 0.0

    # "Saved" cost = what the prompt+completion tokens on cache hits *would*
    # have cost had they not been served from cache.
    saved = sum(
        estimate_cost_usd(l.model, l.prompt_tokens, l.completion_tokens)
        for l in logs
        if l.cache_status != "miss"
    )

    hit_rate = (exact_hits + semantic_hits) / total_requests if total_requests else 0.0

    return UsageSummary(
        client_id=client.id,
        client_name=client.name,
        total_requests=total_requests,
        exact_hits=exact_hits,
        semantic_hits=semantic_hits,
        misses=misses,
        cache_hit_rate=round(hit_rate, 4),
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
        estimated_cost_saved_usd=round(saved, 6),
        avg_latency_ms=round(avg_latency, 2),
    )


@router.get("/usage/{client_id}", response_model=UsageSummary)
def usage_for_client(client_id: int, db: Session = Depends(get_db)):
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return _summarize(db, client)


@router.get("/usage", response_model=list[UsageSummary])
def usage_all_clients(db: Session = Depends(get_db)):
    clients = db.query(Client).all()
    return [_summarize(db, c) for c in clients]


@router.get("/timeseries/{client_id}")
def usage_timeseries(client_id: int, db: Session = Depends(get_db)):
    """Requests-per-day and tokens-per-day for a simple dashboard chart."""
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    rows = (
        db.query(
            func.date(RequestLog.created_at).label("day"),
            func.count(RequestLog.id).label("requests"),
            func.sum(RequestLog.total_tokens).label("tokens"),
            func.sum(RequestLog.estimated_cost_usd).label("cost_usd"),
        )
        .filter(RequestLog.client_id == client_id)
        .group_by(func.date(RequestLog.created_at))
        .order_by(func.date(RequestLog.created_at))
        .all()
    )

    return [
        {
            "day": str(r.day),
            "requests": r.requests,
            "tokens": int(r.tokens or 0),
            "cost_usd": round(float(r.cost_usd or 0), 6),
        }
        for r in rows
    ]
