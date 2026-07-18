"""Per-client rate limiting using a fixed-window counter in Redis.

Each client gets a counter keyed by `ratelimit:{client_id}:{minute_bucket}`
that is incremented on every request and expires automatically after 60s.
This is intentionally simple (fixed window rather than sliding log) to keep
it O(1) per request with a single Redis round trip, which matters since this
runs on the hot path of every proxied call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import redis


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int


class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def check(self, client_id: int, limit_per_minute: int) -> RateLimitResult:
        bucket = int(time.time() // 60)
        key = f"ratelimit:{client_id}:{bucket}"

        count = self.redis.incr(key)
        if count == 1:
            self.redis.expire(key, 60)

        remaining = max(0, limit_per_minute - count)
        reset_seconds = 60 - int(time.time() % 60)

        return RateLimitResult(
            allowed=count <= limit_per_minute,
            limit=limit_per_minute,
            remaining=remaining,
            reset_seconds=reset_seconds,
        )
