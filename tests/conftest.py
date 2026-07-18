import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.embeddings import LocalHashEmbedder
from app.llm_client import MockLLMClient
from app.main import app
from app.rate_limiter import RateLimiter
from app.semantic_cache import SemanticCache


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Enter the TestClient context first so the real lifespan/startup runs
    # (harmless real-settings init against sqlite/lazy-redis), then swap in
    # fakes on app.state so no test ever touches a real Redis/Postgres server.
    with TestClient(app) as test_client:
        fake_redis = fakeredis.FakeRedis(decode_responses=True)
        embedder = LocalHashEmbedder(dim=256)
        app.state.redis = fake_redis
        app.state.semantic_cache = SemanticCache(
            embedder=embedder,
            redis_client=fake_redis,
            similarity_threshold=0.85,
            exact_ttl_seconds=86400,
        )
        app.state.rate_limiter = RateLimiter(fake_redis)
        app.state.llm_client = MockLLMClient()
        app.state.cache_entries_loaded = 0

        yield test_client

    app.dependency_overrides.clear()
