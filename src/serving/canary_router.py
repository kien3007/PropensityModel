"""
Canary Deployment & A/B Testing Router Module.
Manages deterministic Champion vs Challenger traffic routing (e.g. 90/10 split),
online experimentation, and dynamic traffic reallocation.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.models.predictor import PropensityPredictor

logger = logging.getLogger("CanaryRouter")


class CanaryRouter:
    """
    Routes inference traffic between Champion and Challenger models based on deterministic household hash.
    """

    def __init__(
        self,
        champion_predictor: Optional[PropensityPredictor] = None,
        challenger_model_path: Optional[Path | str] = None,
        challenger_traffic_ratio: float = 0.10,  # 10% to Challenger by default
        enabled: bool = False,
    ) -> None:
        self.champion = champion_predictor or PropensityPredictor()
        self.challenger: Optional[PropensityPredictor] = None
        self.traffic_ratio = max(0.0, min(1.0, challenger_traffic_ratio))
        self.enabled = enabled

        if challenger_model_path and Path(challenger_model_path).exists():
            self.load_challenger(challenger_model_path)

    def load_challenger(self, model_path: Path | str) -> bool:
        """Loads a challenger candidate model bundle into memory."""
        try:
            challenger = PropensityPredictor(model_path=model_path)
            champion_def = (self.champion.target, self.champion.feature_set_version, self.champion.horizon_days)
            challenger_def = (challenger.target, challenger.feature_set_version, challenger.horizon_days)
            if challenger_def != champion_def:
                raise ValueError(f"challenger predicts {challenger_def}, champion predicts {champion_def}")
            self.challenger = challenger
            self.enabled = True
            logger.info(
                f"Loaded Challenger model '{self.challenger.bundle_name}' from {model_path}. "
                f"Canary routing active with {int(self.traffic_ratio * 100)}% traffic."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load Challenger model from {model_path}: {e}")
            self.challenger = None
            self.enabled = False
            return False

    def route(self, household_key: int) -> Tuple[PropensityPredictor, str]:
        """
        Deterministically selects Champion or Challenger based on household_key hash.
        Returns (selected_predictor, variant_name: 'champion' | 'challenger').
        """
        if not self.enabled or self.challenger is None or self.traffic_ratio <= 0.0:
            return self.champion, "champion"

        # Deterministic hashing (0-99 bucket)
        hash_digest = hashlib.md5(f"canary_hh_{household_key}".encode("utf-8")).hexdigest()
        bucket = int(hash_digest[:4], 16) % 100

        threshold = int(self.traffic_ratio * 100)
        if bucket < threshold:
            return self.challenger, "challenger"
        return self.champion, "champion"

    def set_traffic_split(self, challenger_ratio: float, enabled: Optional[bool] = None) -> Dict[str, Any]:
        """Updates traffic ratio dynamically."""
        self.traffic_ratio = max(0.0, min(1.0, challenger_ratio))
        if enabled is not None:
            self.enabled = enabled
        logger.info(f"Canary split updated: Challenger={int(self.traffic_ratio * 100)}%, Enabled={self.enabled}")
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Returns current Canary router status and configuration."""
        challenger_pct = int(self.traffic_ratio * 100) if self.enabled and self.challenger else 0
        champion_pct = 100 - challenger_pct
        return {
            "canary_enabled": self.enabled and (self.challenger is not None),
            "traffic_split": {
                "champion_percentage": champion_pct,
                "challenger_percentage": challenger_pct,
            },
            "champion": {
                "bundle_name": self.champion.bundle_name,
                "feature_count": len(self.champion.features),
            },
            "challenger": {
                "bundle_name": self.challenger.bundle_name if self.challenger else None,
                "feature_count": len(self.challenger.features) if self.challenger else None,
                "is_loaded": self.challenger is not None,
            },
        }
