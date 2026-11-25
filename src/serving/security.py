"""
Serving API security: API-key authentication and a per-client rate limiter.

- Write / admin / bulk-export endpoints require an `X-API-Key` header matching one of
  settings.SERVING_API_KEYS (comma-separated). With no keys configured the API is open in
  development (logged once) and fails closed (503) in production.
- RateLimiter is an in-process sliding window (settings.RATE_LIMIT_PER_MINUTE per client and
  route). It protects a single replica; a shared limit across replicas belongs in the gateway.
"""
from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Set

from fastapi import Header, HTTPException, Request, status

from src.config.settings import settings

logger = logging.getLogger("ServingSecurity")
_warned_open = False


def configured_api_keys() -> Set[str]:
    return {k.strip() for k in settings.SERVING_API_KEYS.split(",") if k.strip()}


def admin_request_headers() -> Dict[str, str]:
    """Headers for internal admin calls (scheduler / DAG hot-reload) to the serving API."""
    headers = {"Content-Type": "application/json"}
    keys = sorted(configured_api_keys())
    if keys:
        headers["X-API-Key"] = keys[0]
    return headers


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    global _warned_open
    keys = configured_api_keys()
    if not keys:
        if settings.ENV.lower() == "production":
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API keys are not configured (SERVING_API_KEYS)")
        if not _warned_open:
            logger.warning("SERVING_API_KEYS is empty: protected endpoints are OPEN (development mode).")
            _warned_open = True
        return
    if x_api_key is None or not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key", {"WWW-Authenticate": "ApiKey"})


class RateLimiter:
    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_id: str, limit: int) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client_id]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, int(self.window - (now - hits[0])))
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded",
                                    {"Retry-After": str(retry_after)})
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


async def rate_limit(request: Request) -> None:
    # async: runs on the event loop (no threadpool hop); the check is an in-memory counter.
    client = request.client.host if request.client else "unknown"
    rate_limiter.check(f"{client}:{request.url.path}", settings.RATE_LIMIT_PER_MINUTE)
