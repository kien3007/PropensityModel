"""
Shared Redis connection factory for the speed layer and the serving batch store.

Probes once with retries disabled (redis-py's default retry policy otherwise stalls ~25s
on an unreachable server). Returns None when Redis is unavailable so callers can use an
in-memory fallback, unless settings.STRICT_INFRA is set, in which case it raises.
"""
from __future__ import annotations

import logging
from typing import Optional

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from src.config.settings import settings

logger = logging.getLogger(__name__)


class InfrastructureUnavailable(RuntimeError):
    """Raised in STRICT_INFRA mode when a required backing service is unreachable."""


def connect_redis(component: str) -> Optional[redis.Redis]:
    host = settings.REDIS_HOST
    if host in ("localhost", "0.0.0.0"):
        host = "127.0.0.1"
    kwargs = dict(
        host=host,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
    )
    try:
        probe = redis.Redis(retry=Retry(NoBackoff(), 0), **kwargs)
        probe.ping()
        probe.close()
        logger.info(f"{component}: connected to Redis at {host}:{settings.REDIS_PORT}.")
        return redis.Redis(**kwargs)
    except Exception as exc:
        if settings.STRICT_INFRA:
            raise InfrastructureUnavailable(f"{component}: Redis unreachable at {host}:{settings.REDIS_PORT} ({exc})") from exc
        logger.warning(f"{component}: Redis not available ({exc}); using in-memory fallback (STRICT_INFRA=false).")
        return None
