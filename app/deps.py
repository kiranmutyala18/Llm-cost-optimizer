from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Client
from app.rate_limiter import RateLimiter
from app.semantic_cache import SemanticCache


def get_semantic_cache(request: Request) -> SemanticCache:
    return request.app.state.semantic_cache


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_llm_client(request: Request):
    return request.app.state.llm_client


def get_current_client(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Client:
    client = db.query(Client).filter(Client.api_key == x_api_key).first()
    if not client:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not client.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Client is deactivated")
    return client
