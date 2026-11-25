"""
Serving layer: answers propensity requests from the precomputed weekly batch view.

- The batch view comes from the shared BatchScoreStore, for the variant the canary router
  assigns (champion / challenger). If the challenger has no batch scores, the champion's
  view is served and reported as such.
- Households without a batch view get an explicit cold-start answer (population mean
  probability, score_source="cold_start") instead of a silent 0.5.

The served probability is exactly what the batch layer computed with the champion model, so
what is monitored (drift, realized AUC, model card) is what clients receive.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.models.predictor import PropensityPredictor
from src.serving.batch_store import BatchScoreStore
from src.serving.canary_router import CanaryRouter

logger = logging.getLogger(__name__)


class PropensityScoringService:
    def __init__(
        self,
        predictor: Optional[PropensityPredictor] = None,
        canary_router: Optional[CanaryRouter] = None,
        batch_store: Optional[BatchScoreStore] = None,
    ) -> None:
        self.predictor = predictor or PropensityPredictor(expected_target="purchase")
        self.router = canary_router or CanaryRouter(champion_predictor=self.predictor)
        self.batch_store = batch_store or BatchScoreStore()

    def reload_predictor(self) -> None:
        """Hot-reloads the champion bundle (metadata for /v1/models/champion and cold start)."""
        self.predictor.reload()
        self.router.champion = self.predictor

    def set_batch_cache(self, batch_df: pd.DataFrame, variant: str = "champion") -> int:
        """Publishes batch scores into the shared store (used by tests and ad-hoc loads)."""
        return self.batch_store.publish(batch_df, variant, {"source": "set_batch_cache"})

    def cold_start_probability(self) -> float:
        meta = self.batch_store.metadata("champion") or {}
        if meta.get("mean_probability") is not None:
            return float(meta["mean_probability"])
        base_rate = (self.predictor.bundle.get("metrics") or {}).get("base_rate")
        return float(base_rate) if base_rate is not None else 0.5

    def get_propensity(self, household_key: int) -> Dict[str, Any]:
        _, routed_variant = self.router.route(household_key)

        variant = routed_variant
        view = self.batch_store.get(household_key, variant)
        if view is None and variant == "challenger":
            variant = "champion"
            view = self.batch_store.get(household_key, variant)

        if view is None:
            score_source = "cold_start"
            probability = round(self.cold_start_probability(), 4)
            decile, urgency, action = None, "UNKNOWN", "COLD_START_DEFAULT"
        else:
            score_source = "batch"
            probability = float(view["batch_probability"])
            decile, urgency, action = int(view["decile"]), view["urgency_tier"], view["marketing_action"]

        meta = self.batch_store.metadata(variant) or {}
        return {
            "household_key": household_key,
            "propensity": probability,
            "decile": decile,
            "urgency_tier": urgency,
            "marketing_action": action,
            "model_variant": variant,
            "routed_variant": routed_variant,
            "score_source": score_source,
            "batch_snapshot_day": meta.get("snapshot_day"),
        }
