"""
Speed Layer: Real-Time Stream & Intraday Delta Processor.
Tracks intraday activity (POS receipts, coupon clips, app logins) in Redis with 24h TTL.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
import redis

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SpeedLayerProcessor:
    """
    Manages intraday streaming event ingestion and real-time delta state.
    """

    def __init__(self) -> None:
        self.redis_client: Optional[redis.Redis] = None
        self._memory_fallback: Dict[int, Dict[str, Any]] = {}
        self._init_redis()

    def _init_redis(self) -> None:
        host = settings.REDIS_HOST
        if host in ("localhost", "0.0.0.0"):
            host = "127.0.0.1"
        try:
            r = redis.Redis(
                host=host,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=0.5,
            )
            r.ping()
            self.redis_client = r
            logger.info("SpeedLayer connected to Redis successfully.")
        except Exception as e:
            logger.warning(f"Redis not available ({e}). Using in-memory fallback for SpeedLayer.")
            self.redis_client = None

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes an intraday streaming event:
        - pos_checkout: household completed a purchase today.
        - coupon_clip: household saved a discount coupon on app.
        - app_open: household opened mobile app or viewed products.
        """
        hh_key = int(event.get("household_key", 0))
        event_type = event.get("event_type", "unknown")
        now = int(time.time())

        delta = self.get_delta(hh_key)
        delta["updated_at"] = now

        if event_type == "pos_checkout":
            delta["intraday_trips_detected"] = 1
            delta["intraday_spend"] += float(event.get("sales_value", 0.0))
        elif event_type == "coupon_clip":
            delta["intraday_coupons_clipped"] += 1
        elif event_type == "app_open":
            delta["intraday_app_interactions"] += 1

        self._save_delta(hh_key, delta)
        logger.info(f"SpeedLayer updated delta for household {hh_key}: {event_type}")
        return delta

    def get_delta(self, household_key: int) -> Dict[str, Any]:
        """
        Retrieves the intraday delta vector for a household.
        """
        default_delta = {
            "household_key": household_key,
            "intraday_trips_detected": 0,
            "intraday_coupons_clipped": 0,
            "intraday_spend": 0.0,
            "intraday_app_interactions": 0,
            "updated_at": 0,
        }

        if self.redis_client:
            try:
                raw = self.redis_client.get(f"realtime:delta:{household_key}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.error(f"Error reading from Redis: {e}")

        return self._memory_fallback.get(household_key, default_delta)

    def _save_delta(self, household_key: int, delta: Dict[str, Any]) -> None:
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"realtime:delta:{household_key}",
                    settings.REDIS_TTL_SECONDS,
                    json.dumps(delta),
                )
                return
            except Exception as e:
                logger.error(f"Error writing to Redis: {e}")

        self._memory_fallback[household_key] = delta

    def reset_intraday_deltas(self) -> None:
        """
        Resets intraday delta views after Batch Layer completes master ingestion.
        """
        if self.redis_client:
            try:
                keys = self.redis_client.keys("realtime:delta:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} intraday delta keys from Redis.")
            except Exception as e:
                logger.error(f"Error resetting Redis keys: {e}")

        self._memory_fallback.clear()
