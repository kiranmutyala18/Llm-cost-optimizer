"""Two-tier cache for LLM responses.

Tier 1 (fast path): exact-match cache in Redis, keyed by a hash of the
normalized prompt. O(1) lookup, sub-millisecond.

Tier 2 (semantic path): a FAISS in-memory index over prompt embeddings.
On a Tier-1 miss, we search FAISS for the nearest neighbor; if its cosine
similarity is above `similarity_threshold`, we treat it as a cache hit and
return the previously-generated response instead of calling the LLM again.

The FAISS index is rebuilt from the `cache_entries` table in Postgres on
startup, and every new entry is written to Postgres *and* added to the live
index, so the cache survives restarts without needing FAISS's own
(less convenient) persistence format.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import faiss
import numpy as np
import redis
from sqlalchemy.orm import Session

from app.embeddings import BaseEmbedder
from app.models import CacheEntry


def normalize_prompt(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise so that
    trivially-different phrasings hash the same for the exact-match tier."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


@dataclass
class CacheLookupResult:
    hit: bool
    status: str  # "exact_hit" | "semantic_hit" | "miss"
    response_text: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    similarity_score: float | None = None
    cache_entry_id: str | None = None


class SemanticCache:
    def __init__(
        self,
        embedder: BaseEmbedder,
        redis_client: redis.Redis,
        similarity_threshold: float = 0.92,
        exact_ttl_seconds: int = 86400,
    ):
        self.embedder = embedder
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.exact_ttl_seconds = exact_ttl_seconds

        # Inner-product index over L2-normalized vectors == cosine similarity.
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedder.dim))
        # Map FAISS int64 ids -> cache_entries.id (str uuid), since FAISS needs ints.
        self._id_map: dict[int, str] = {}
        self._next_faiss_id = 0

    # ---------- index bootstrap ----------

    def load_from_db(self, db: Session) -> int:
        """Rebuild the in-memory FAISS index from persisted cache entries.
        Call once at application startup."""
        entries = db.query(CacheEntry).all()
        if not entries:
            return 0

        vectors = np.array([e.embedding for e in entries], dtype=np.float32)
        faiss_ids = np.arange(self._next_faiss_id, self._next_faiss_id + len(entries), dtype=np.int64)
        self.index.add_with_ids(vectors, faiss_ids)

        for fid, entry in zip(faiss_ids.tolist(), entries):
            self._id_map[fid] = entry.id

        self._next_faiss_id += len(entries)
        return len(entries)

    # ---------- lookup ----------

    def lookup(self, db: Session, raw_prompt: str) -> CacheLookupResult:
        norm = normalize_prompt(raw_prompt)
        phash = prompt_hash(raw_prompt)

        # Tier 1: exact match in Redis
        redis_key = f"exact:{phash}"
        cached = self.redis.get(redis_key)
        if cached:
            entry = db.query(CacheEntry).filter(CacheEntry.prompt_hash == phash).first()
            if entry:
                self._record_hit(db, entry)
                return CacheLookupResult(
                    hit=True,
                    status="exact_hit",
                    response_text=entry.response_text,
                    prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens,
                    similarity_score=1.0,
                    cache_entry_id=entry.id,
                )

        # Tier 2: semantic match via FAISS
        if self.index.ntotal > 0:
            query_vec = self.embedder.embed(norm).reshape(1, -1).astype(np.float32)
            scores, ids = self.index.search(query_vec, k=1)
            best_score = float(scores[0][0])
            best_id = int(ids[0][0])
            if best_id != -1 and best_score >= self.similarity_threshold:
                entry_id = self._id_map.get(best_id)
                entry = db.get(CacheEntry, entry_id) if entry_id else None
                if entry:
                    self._record_hit(db, entry)
                    # Warm the exact-match tier too, so identical repeats are O(1) next time.
                    self.redis.setex(redis_key, self.exact_ttl_seconds, "1")
                    return CacheLookupResult(
                        hit=True,
                        status="semantic_hit",
                        response_text=entry.response_text,
                        prompt_tokens=entry.prompt_tokens,
                        completion_tokens=entry.completion_tokens,
                        similarity_score=best_score,
                        cache_entry_id=entry.id,
                    )

        return CacheLookupResult(hit=False, status="miss")

    def _record_hit(self, db: Session, entry: CacheEntry) -> None:
        entry.hit_count += 1
        entry.last_hit_at = datetime.now(timezone.utc)
        db.add(entry)
        db.commit()

    # ---------- write-through on miss ----------

    def store(
        self,
        db: Session,
        raw_prompt: str,
        response_text: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CacheEntry:
        norm = normalize_prompt(raw_prompt)
        phash = prompt_hash(raw_prompt)
        vec = self.embedder.embed(norm)

        entry = CacheEntry(
            prompt_normalized=norm,
            prompt_hash=phash,
            response_text=response_text,
            embedding=vec.tolist(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)

        faiss_id = self._next_faiss_id
        self._next_faiss_id += 1
        self.index.add_with_ids(vec.reshape(1, -1).astype(np.float32), np.array([faiss_id], dtype=np.int64))
        self._id_map[faiss_id] = entry.id

        self.redis.setex(f"exact:{phash}", self.exact_ttl_seconds, "1")
        return entry
