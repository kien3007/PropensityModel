"""
Batch View Store: precomputed batch scores shared between the batch layer (writer) and
every serving replica (readers).

Redis layout per variant ("champion" | "challenger"):
  batch:<variant>:scores  hash  household_key -> JSON batch view
  batch:<variant>:meta    JSON  run_id, snapshot_day, model_bundle, households, mean_probability
A publish writes a staging hash and RENAMEs it over the live key, so readers switch from
the old to the new snapshot atomically and never see a half-written batch.

Without Redis (STRICT_INFRA=false) the store is in-process; the API then bootstraps it from
the newest batch manifest in settings.OUTPUT_DIR at start-up.
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.redis_client import connect_redis

logger = logging.getLogger(__name__)

VARIANTS = ("champion", "challenger")
VIEW_COLUMNS = ["household_key", "batch_probability", "rank", "decile", "urgency_tier", "marketing_action"]
_UNSET = object()


def _scores_key(variant: str) -> str:
    return f"batch:{variant}:scores"


def _meta_key(variant: str) -> str:
    return f"batch:{variant}:meta"


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'; expected one of {VARIANTS}")


class BatchScoreStore:
    def __init__(self, redis_client: Any = _UNSET) -> None:
        self.redis = connect_redis("BatchScoreStore") if redis_client is _UNSET else redis_client
        self._memory: Dict[str, Dict[int, Dict[str, Any]]] = {v: {} for v in VARIANTS}
        self._meta: Dict[str, Optional[Dict[str, Any]]] = {v: None for v in VARIANTS}

    @property
    def backend(self) -> str:
        return "redis" if self.redis else "memory"

    # ----------------------------------------------------------------- writes
    def publish(self, scores: pd.DataFrame, variant: str = "champion", metadata: Optional[Dict[str, Any]] = None) -> int:
        _check_variant(variant)
        view = scores.rename(columns={"purchase_probability": "batch_probability"})
        missing = [c for c in VIEW_COLUMNS if c not in view.columns]
        if missing:
            raise ValueError(f"Batch scores are missing columns {missing}")
        view = view[VIEW_COLUMNS].astype({"household_key": int, "batch_probability": float, "rank": int, "decile": int})
        records = view.to_dict("records")
        meta = {
            "households": len(records),
            "mean_probability": float(view["batch_probability"].mean()) if records else None,
            "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            **(metadata or {}),
        }

        if self.redis:
            staging = f"{_scores_key(variant)}:staging:{uuid.uuid4().hex}"
            pipe = self.redis.pipeline(transaction=False)
            for start in range(0, len(records), 1000):
                chunk = records[start:start + 1000]
                pipe.hset(staging, mapping={str(r["household_key"]): json.dumps(r) for r in chunk})
            pipe.execute()
            tx = self.redis.pipeline(transaction=True)
            tx.rename(staging, _scores_key(variant))
            tx.set(_meta_key(variant), json.dumps(meta))
            tx.execute()
        else:
            self._memory[variant] = {r["household_key"]: r for r in records}
            self._meta[variant] = meta
        logger.info(f"Published {len(records):,} '{variant}' batch views ({self.backend}).")
        return len(records)

    def clear(self, variant: Optional[str] = None) -> None:
        for v in ([variant] if variant else VARIANTS):
            if self.redis:
                self.redis.delete(_scores_key(v), _meta_key(v))
            self._memory[v] = {}
            self._meta[v] = None

    # ------------------------------------------------------------------ reads
    def get(self, household_key: int, variant: str = "champion") -> Optional[Dict[str, Any]]:
        if self.redis:
            raw = self.redis.hget(_scores_key(variant), str(household_key))
            return json.loads(raw) if raw else None
        record = self._memory[variant].get(int(household_key))
        return dict(record) if record else None

    def metadata(self, variant: str = "champion") -> Optional[Dict[str, Any]]:
        if self.redis:
            raw = self.redis.get(_meta_key(variant))
            return json.loads(raw) if raw else None
        return self._meta[variant]

    def count(self, variant: str = "champion") -> int:
        if self.redis:
            return int(self.redis.hlen(_scores_key(variant)))
        return len(self._memory[variant])

    def records(self, variant: str = "champion", decile: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if self.redis:
            rows = [json.loads(v) for v in self.redis.hvals(_scores_key(variant))]
        else:
            rows = list(self._memory[variant].values())
        if decile is not None:
            rows = [r for r in rows if r["decile"] == decile]
        return sorted(rows, key=lambda r: r["rank"])[:limit]

    # -------------------------------------------------------------- bootstrap
    def bootstrap_from_outputs(self, output_dir: Path | str) -> bool:
        """Loads the newest batch run's parquet artefacts if the store holds no champion scores."""
        if self.count("champion") > 0:
            return False
        manifests = sorted(Path(output_dir).glob("batch_manifest_day_*.json"), key=lambda p: p.stat().st_mtime)
        if not manifests:
            logger.warning(f"No batch manifest in {output_dir}; serving will answer cold-start until a batch runs.")
            return False
        manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})
        loaded = False
        for variant, key in (("champion", "parquet"), ("challenger", "challenger_parquet")):
            path = artifacts.get(key)
            if path and Path(path).exists():
                meta = {"run_id": manifest.get("run_id"), "snapshot_day": manifest.get("cutoff_day"),
                        "model_bundle": manifest.get("model_bundle" if variant == "champion" else "challenger_bundle"),
                        "source": "bootstrap"}
                self.publish(pd.read_parquet(path), variant, meta)
                loaded = True
        return loaded
