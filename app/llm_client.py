"""Abstraction over the actual "upstream" LLM being proxied.

- MockLLMClient: deterministic, offline, zero-cost. Used for local dev/tests
  and any environment without outbound network access. It still produces a
  realistic token count via tiktoken so the rest of the pipeline (cost
  tracking, analytics) can be exercised meaningfully.

- OpenAILLMClient: forwards the request to a real OpenAI-compatible
  /chat/completions endpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import tiktoken

from app.config import Settings

# Rough public per-1K-token pricing (USD) used only for cost *estimation* in
# analytics. Update as needed -- this is not billing-grade, just a useful
# dashboard signal for "how much am I saving with caching."
PRICING_PER_1K_TOKENS = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "default": {"prompt": 0.0005, "completion": 0.0015},
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["default"])
    return (prompt_tokens / 1000) * rates["prompt"] + (completion_tokens / 1000) * rates["completion"]


def _approx_token_count(text: str) -> int:
    """Fallback token estimate (~4 chars/token, the common rule of thumb)
    used only if tiktoken's BPE files can't be fetched (e.g. no outbound
    network access to openaipublic.blob.core.windows.net)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # tiktoken needs to download its BPE merge file on first use; if
        # that network call fails (offline/sandboxed environment), fall
        # back to a cheap approximation rather than erroring the request.
        return _approx_token_count(text)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, model: str, messages: list[dict], temperature: float = 0.7) -> LLMResult:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Offline stand-in for a real LLM. Echoes a deterministic, plausible
    completion so caching/analytics logic can be fully tested without
    network access or an API key."""

    def complete(self, model: str, messages: list[dict], temperature: float = 0.7) -> LLMResult:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        prompt_text = "\n".join(m["content"] for m in messages)
        completion_text = f"[mock-{model}] Here is a response to: {last_user[:200]}"

        prompt_tokens = count_tokens(prompt_text, model)
        completion_tokens = count_tokens(completion_text, model)

        return LLMResult(text=completion_text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str):
        import httpx

        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    def complete(self, model: str, messages: list[dict], temperature: float = 0.7) -> LLMResult:
        resp = self._client.post(
            "/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResult(
            text=choice,
            prompt_tokens=usage.get("prompt_tokens", count_tokens("\n".join(m["content"] for m in messages), model)),
            completion_tokens=usage.get("completion_tokens", count_tokens(choice, model)),
        )


def build_llm_client(settings: Settings) -> BaseLLMClient:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set")
        return OpenAILLMClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return MockLLMClient()
