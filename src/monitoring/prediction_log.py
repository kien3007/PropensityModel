"""
Served-prediction log: what the API answered, with which fusion rules, for later
comparison against realized purchases (fusion-rule calibration, Phase 5.4).

Redis stream `monitoring:served_predictions` (capped at settings.PREDICTION_LOG_MAXLEN),
or an in-process ring buffer without Redis. Appending never fails a request.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any, Deque, Dict

import pandas as pd

from src.common.redis_client import connect_redis
from src.config.settings import settings

logger = logging.getLogger(__name__)
STREAM_KEY = "monitoring:served_predictions"
_UNSET = object()


class PredictionLog:
    def __init__(self, redis_client: Any = _UNSET) -> None:
        self.redis = connect_redis("PredictionLog") if redis_client is _UNSET else redis_client
        self._memory: Deque[Dict[str, Any]] = deque(maxlen=settings.PREDICTION_LOG_MAXLEN)

    def append(self, result: Dict[str, Any]) -> None:
        record = {
            "served_at": time.time(),
            "household_key": result["household_key"],
            "batch_snapshot_day": result.get("batch_snapshot_day"),
            "model_variant": result.get("model_variant"),
            "score_source": result.get("score_source"),
            "batch_probability": result["batch_baseline_propensity"],
            "unified_probability": result["unified_propensity"],
            "rules": result.get("applied_fusion_rules", []),
        }
        try:
            if self.redis:
                self.redis.xadd(STREAM_KEY, {"r": json.dumps(record)}, maxlen=settings.PREDICTION_LOG_MAXLEN,
                                approximate=True)
            else:
                self._memory.append(record)
        except Exception as e:
            logger.error(f"Prediction log append failed: {e}")

    def to_frame(self) -> pd.DataFrame:
        if self.redis:
            rows = [json.loads(fields["r"]) for _, fields in self.redis.xrange(STREAM_KEY)]
        else:
            rows = list(self._memory)
        return pd.DataFrame(rows)

    def clear(self) -> None:
        if self.redis:
            self.redis.delete(STREAM_KEY)
        self._memory.clear()
