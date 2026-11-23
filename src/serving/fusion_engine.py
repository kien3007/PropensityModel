"""
Serving Layer: Lambda Fusion Engine.

P_final = Merge(BatchView[variant][h], RealTimeDelta[h])

- The batch view comes from the shared BatchScoreStore, for the variant the canary router
  assigns (champion / challenger). If the challenger has no batch scores, the champion's
  view is served and reported as such.
- Households without a batch view get an explicit cold-start answer (population mean
  probability, score_source="cold_start") instead of a silent 0.5.
- Fusion rules are HEURISTIC adjustments configured in settings (FUSION_*); they are not
  learned and should be recalibrated from ground-truth feedback.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.config.settings import settings
from src.models.predictor import PropensityPredictor
from src.serving.batch_store import BatchScoreStore
from src.serving.canary_router import CanaryRouter
from src.speed_layer.delta_stream import SpeedLayerProcessor

logger = logging.getLogger(__name__)


class LambdaFusionEngine:
    def __init__(
        self,
        predictor: Optional[PropensityPredictor] = None,
        speed_processor: Optional[SpeedLayerProcessor] = None,
        canary_router: Optional[CanaryRouter] = None,
        batch_store: Optional[BatchScoreStore] = None,
    ) -> None:
        self.predictor = predictor or PropensityPredictor(expected_target="purchase")
        self.speed_processor = speed_processor or SpeedLayerProcessor()
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

    def get_unified_propensity(self, household_key: int) -> Dict[str, Any]:
        _, routed_variant = self.router.route(household_key)

        variant = routed_variant
        view = self.batch_store.get(household_key, variant)
        if view is None and variant == "challenger":
            variant = "champion"
            view = self.batch_store.get(household_key, variant)

        if view is None:
            score_source = "cold_start"
            base_p = round(self.cold_start_probability(), 4)
            decile, urgency, action = None, "UNKNOWN", "COLD_START_DEFAULT"
        else:
            score_source = "batch"
            base_p = float(view["batch_probability"])
            decile, urgency, action = int(view["decile"]), view["urgency_tier"], view["marketing_action"]

        delta = self.speed_processor.get_delta(household_key)
        unified_p = base_p
        applied_rules = []

        # Rule 1: intraday checkout detected -> already shopped today
        if delta.get("intraday_trips_detected", 0) == 1:
            unified_p = round(base_p * settings.FUSION_PURCHASE_DAMPENING, 4)
            urgency, action = "LOW", "POST_PURCHASE_COOLDOWN"
            applied_rules.append("intraday_purchase_dampening")
        # Rule 2: coupon clipped today -> intent surge
        elif delta.get("intraday_coupons_clipped", 0) >= 1:
            boost = settings.FUSION_COUPON_BOOST_PER_CLIP * delta["intraday_coupons_clipped"]
            unified_p = round(min(1.0, base_p + boost), 4)
            urgency, action = "URGENT", "VIP_COUPON_HALO_UPSELL"
            applied_rules.append("coupon_clipping_intent_surge")
        # Rule 3: repeated app interactions -> slight engagement lift
        elif delta.get("intraday_app_interactions", 0) >= settings.FUSION_APP_MIN_INTERACTIONS:
            unified_p = round(min(1.0, base_p + settings.FUSION_APP_NUDGE), 4)
            applied_rules.append("app_browsing_nudge")

        meta = self.batch_store.metadata(variant) or {}
        return {
            "household_key": household_key,
            "unified_propensity": unified_p,
            "batch_baseline_propensity": base_p,
            "decile": decile,
            "urgency_tier": urgency,
            "marketing_action": action,
            "intraday_delta": delta,
            "applied_fusion_rules": applied_rules,
            "model_variant": variant,
            "routed_variant": routed_variant,
            "score_source": score_source,
            "batch_snapshot_day": meta.get("snapshot_day"),
        }
