"""
Speed Layer: Real-Time Stream & Intraday Delta Processor.

Tracks intraday activity (POS checkouts, coupon clips, app opens) per household in Redis
(key realtime:delta:{hh}, TTL settings.REDIS_TTL_SECONDS). Processing is idempotent:
an event carrying an `event_id` that was already applied (Kafka redelivery, client retry)
is acknowledged without changing the delta.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Set

import redis

from src.common.redis_client import connect_redis
from src.config.settings import settings

logger = logging.getLogger(__name__)

DELTA_KEY = "realtime:delta:{hh}"
SEEN_KEY = "realtime:seen:{event_id}"


def _empty_delta(household_key: int) -> Dict[str, Any]:
    return {
        "household_key": household_key,
        "intraday_trips_detected": 0,
        "intraday_coupons_clipped": 0,
        "intraday_spend": 0.0,
        "intraday_app_interactions": 0,
        "updated_at": 0,
    }


class SpeedLayerProcessor:
    def __init__(self) -> None:
        self.redis_client: Optional[redis.Redis] = connect_redis("SpeedLayer")
        self._memory_fallback: Dict[int, Dict[str, Any]] = {}
        self._memory_seen: Set[str] = set()

    @property
    def backend(self) -> str:
        return "redis" if self.redis_client else "memory"

    def active_delta_count(self) -> int:
        if self.redis_client:
            try:
                return sum(1 for _ in self.redis_client.scan_iter(DELTA_KEY.format(hh="*")))
            except Exception as e:
                logger.error(f"Error counting Redis deltas: {e}")
        return len(self._memory_fallback)

    def _first_time_seen(self, event_id: Optional[str]) -> bool:
        """Marks event_id as applied; False if it was applied before."""
        if not event_id:
            return True
        if self.redis_client:
            try:
                key = SEEN_KEY.format(event_id=event_id)
                return bool(self.redis_client.set(key, 1, nx=True, ex=settings.EVENT_DEDUP_TTL_SECONDS))
            except Exception as e:
                logger.error(f"Error checking event idempotency in Redis: {e}")
        if event_id in self._memory_seen:
            return False
        self._memory_seen.add(event_id)
        return True

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies an intraday event to the household's delta:
        - pos_checkout: household completed a purchase today (spend clamped to >= 0)
        - coupon_clip: household saved a coupon
        - app_open: household opened the app / viewed products
        """
        hh_key = int(event.get("household_key", 0))
        event_type = event.get("event_type", "unknown")

        if not self._first_time_seen(event.get("event_id")):
            logger.info(f"SpeedLayer ignored duplicate event {event.get('event_id')} for household {hh_key}")
            return {**self.get_delta(hh_key), "duplicate": True}

        delta = self.get_delta(hh_key)
        delta["updated_at"] = int(time.time())
        if event_type == "pos_checkout":
            delta["intraday_trips_detected"] = 1
            delta["intraday_spend"] += max(0.0, float(event.get("sales_value") or 0.0))
        elif event_type == "coupon_clip":
            delta["intraday_coupons_clipped"] += 1
        elif event_type == "app_open":
            delta["intraday_app_interactions"] += 1

        self._save_delta(hh_key, delta)
        logger.info(f"SpeedLayer updated delta for household {hh_key}: {event_type}")
        return delta

    def get_delta(self, household_key: int) -> Dict[str, Any]:
        if self.redis_client:
            try:
                raw = self.redis_client.get(DELTA_KEY.format(hh=household_key))
                if raw:
                    return json.loads(raw)
                return _empty_delta(household_key)
            except Exception as e:
                logger.error(f"Error reading from Redis: {e}")
        return dict(self._memory_fallback.get(household_key, _empty_delta(household_key)))

    def _save_delta(self, household_key: int, delta: Dict[str, Any]) -> None:
        if self.redis_client:
            try:
                self.redis_client.setex(DELTA_KEY.format(hh=household_key), settings.REDIS_TTL_SECONDS, json.dumps(delta))
                return
            except Exception as e:
                logger.error(f"Error writing to Redis: {e}")
        self._memory_fallback[household_key] = delta

    def reset_intraday_deltas(self) -> None:
        """Clears intraday deltas once the batch layer has absorbed the day's data."""
        if self.redis_client:
            try:
                keys = list(self.redis_client.scan_iter(DELTA_KEY.format(hh="*")))
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} intraday delta keys from Redis.")
            except Exception as e:
                logger.error(f"Error resetting Redis keys: {e}")
        self._memory_fallback.clear()
