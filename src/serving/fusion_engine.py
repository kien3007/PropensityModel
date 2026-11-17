"""
Serving Layer: Unified Lambda Fusion Engine.
Fuses Precomputed Batch Views with Real-Time Speed Layer Deltas.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import pandas as pd

from src.models.predictor import PropensityPredictor
from src.serving.canary_router import CanaryRouter
from src.speed_layer.delta_stream import SpeedLayerProcessor

logger = logging.getLogger(__name__)


class LambdaFusionEngine:
    """
    Implements the Serving Layer Fusion function with A/B Canary Routing:
    P_final = Merge(BatchView[h], RealTimeDelta[h])
    """

    def __init__(
        self,
        predictor: Optional[PropensityPredictor] = None,
        speed_processor: Optional[SpeedLayerProcessor] = None,
        canary_router: Optional[CanaryRouter] = None,
    ) -> None:
        self.predictor = predictor or PropensityPredictor()
        self.speed_processor = speed_processor or SpeedLayerProcessor()
        self.router = canary_router or CanaryRouter(champion_predictor=self.predictor)
        self._batch_cache: Dict[int, Dict[str, Any]] = {}

    def reload_predictor(self) -> None:
        """Hot-reloads champion model in serving gateway."""
        self.predictor.reload()
        self.router.champion = self.predictor

    def set_batch_cache(self, batch_df: pd.DataFrame) -> None:
        """
        Updates the serving cache with precomputed weekly batch scores.
        """
        for _, row in batch_df.iterrows():
            hh = int(row["household_key"])
            self._batch_cache[hh] = {
                "household_key": hh,
                "batch_probability": float(row.get("purchase_probability", 0.5)),
                "rank": int(row.get("rank", 1250)),
                "decile": int(row.get("decile", 5)),
                "urgency_tier": str(row.get("urgency_tier", "REGULAR")),
                "marketing_action": str(row.get("marketing_action", "STANDARD_ENGAGEMENT")),
            }
        logger.info(f"Serving Layer cached batch scores for {len(self._batch_cache)} households.")

    def get_unified_propensity(self, household_key: int) -> Dict[str, Any]:
        """
        Computes unified real-time propensity score:
        Merges precomputed batch baseline with live speed layer delta.
        """
        # 0. Route request via Canary Router (Champion vs Challenger)
        predictor, variant = self.router.route(household_key)

        # 1. Fetch precomputed Batch View
        batch_view = self._batch_cache.get(
            household_key,
            {
                "household_key": household_key,
                "batch_probability": 0.50,
                "rank": 1250,
                "decile": 5,
                "urgency_tier": "REGULAR",
                "marketing_action": "STANDARD_ENGAGEMENT",
            },
        )

        # 2. Fetch live Speed View delta
        delta = self.speed_processor.get_delta(household_key)

        base_p = batch_view["batch_probability"]
        unified_p = base_p
        urgency = batch_view["urgency_tier"]
        action = batch_view["marketing_action"]
        applied_rules = []

        # Rule 1: Intraday checkout detected -> Household already shopped today
        if delta.get("intraday_trips_detected", 0) == 1:
            unified_p = round(base_p * 0.25, 4)
            urgency = "LOW"
            action = "POST_PURCHASE_COOLDOWN"
            applied_rules.append("intraday_purchase_dampening")

        # Rule 2: Active coupon clipped today -> Intent surge
        elif delta.get("intraday_coupons_clipped", 0) >= 1:
            boost = 0.15 * delta["intraday_coupons_clipped"]
            unified_p = round(min(1.0, base_p + boost), 4)
            urgency = "URGENT"
            action = "VIP_COUPON_HALO_UPSELL"
            applied_rules.append("coupon_clipping_intent_surge")

        # Rule 3: Mobile app interactions -> Slight engagement lift
        elif delta.get("intraday_app_interactions", 0) >= 2:
            unified_p = round(min(1.0, base_p + 0.05), 4)
            applied_rules.append("app_browsing_nudge")

        return {
            "household_key": household_key,
            "unified_propensity": unified_p,
            "batch_baseline_propensity": base_p,
            "decile": batch_view["decile"],
            "urgency_tier": urgency,
            "marketing_action": action,
            "intraday_delta": delta,
            "applied_fusion_rules": applied_rules,
            "model_variant": variant,
        }
