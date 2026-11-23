"""
Monitoring State: latest drift / feedback reports shared across processes.

The batch pipeline, feedback jobs and scheduler run in their own processes, but Prometheus
scrapes only the serving API. Producers publish their latest report here (Redis key
monitoring:latest:<kind>, plus a JSON file under OUTPUT_DIR/reports for local runs);
the API reads it when /metrics is scraped and refreshes the gauges.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.redis_client import connect_redis
from src.config.settings import settings

logger = logging.getLogger(__name__)
_UNSET = object()


class MonitoringState:
    def __init__(self, redis_client: Any = _UNSET, report_dir: Optional[Path | str] = None) -> None:
        self.redis = connect_redis("MonitoringState") if redis_client is _UNSET else redis_client
        self.report_dir = Path(report_dir or (settings.OUTPUT_DIR / "reports"))

    def _file(self, kind: str) -> Path:
        return self.report_dir / f"{kind}_latest.json"

    def publish(self, kind: str, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, default=str)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._file(kind).write_text(body, encoding="utf-8")
        if self.redis:
            try:
                self.redis.set(f"monitoring:latest:{kind}", body)
            except Exception as e:
                logger.error(f"Could not publish monitoring state '{kind}' to Redis: {e}")

    def latest(self, kind: str) -> Optional[Dict[str, Any]]:
        if self.redis:
            try:
                raw = self.redis.get(f"monitoring:latest:{kind}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.error(f"Could not read monitoring state '{kind}' from Redis: {e}")
        path = self._file(kind)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
