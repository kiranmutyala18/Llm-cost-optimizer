"""Pluggable text-embedding providers for the semantic cache.

Two implementations are provided:

- LocalHashEmbedder: a fast, free, fully offline embedder based on a hashing
  trick over word n-grams. It has no external dependencies and produces
  stable vectors, which makes it ideal for local dev/tests and for
  environments without outbound network access. It is *not* as semantically
  rich as a trained model, but it correctly places near-duplicate / reworded
  prompts close together, which is enough to demonstrate and exercise the
  caching logic end-to-end.

- OpenAIEmbedder: calls OpenAI's embeddings API for production-quality
  semantic similarity. Swap to this via EMBEDDING_PROVIDER=openai.

Both implementations expose the same `embed(text) -> np.ndarray` interface,
so the rest of the app never needs to know which one is active.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from app.config import Settings


class BaseEmbedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalized embedding vector of shape (self.dim,)."""
        raise NotImplementedError


class LocalHashEmbedder(BaseEmbedder):
    """Deterministic, dependency-free embedder using a hashing trick over
    normalized word unigrams + bigrams. No network calls, no model download.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        words = re.findall(r"[a-z0-9']+", text)
        tokens = list(words)
        tokens += [f"{a}_{b}" for a, b in zip(words, words[1:])]  # bigrams
        return tokens

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class OpenAIEmbedder(BaseEmbedder):
    """Production embedder backed by OpenAI's embeddings API
    (text-embedding-3-small, 1536 dims by default)."""

    def __init__(self, api_key: str, base_url: str, dim: int = 1536,
                 model: str = "text-embedding-3-small"):
        import httpx

        self.dim = dim
        self.model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.post(
            "/embeddings",
            json={"model": self.model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()["data"][0]["embedding"]
        vec = np.array(data, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def build_embedder(settings: Settings) -> BaseEmbedder:
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return LocalHashEmbedder(dim=settings.embedding_dim)
