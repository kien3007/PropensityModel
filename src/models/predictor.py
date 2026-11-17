"""
Champion Model Predictor & Calibration Module.
Loads the calibrated LightGBM bundle, generates purchase probabilities,
and assigns business deciles and marketing action tiers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from src.config.settings import settings
from src.features.pit_extractor import MODEL_FEATURES

logger = logging.getLogger(__name__)


class PropensityPredictor:
    """
    Manages loading the champion model bundle, scoring households,
    and business tier assignment.
    """

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.bundle: Dict[str, Any] = {}
        self.model = None
        self.calibrator = None
        self.features: List[str] = MODEL_FEATURES
        self.epsilon: float = 1e-4
        self.bundle_name: str = "18 + chu kỳ"

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {self.model_path}")

        package = joblib.load(self.model_path)
        self.epsilon = package.get("epsilon", 1e-4)
        bundles = package.get("bundles", {})

        if "28 multi-domain (projectphase3)" in bundles:
            self.bundle_name = "28 multi-domain (projectphase3)"
        elif "18 + chu kỳ" in bundles:
            self.bundle_name = "18 + chu kỳ"
        elif "18 feature" in bundles:
            self.bundle_name = "18 feature"
        else:
            self.bundle_name = list(bundles.keys())[0]

        self.bundle = bundles[self.bundle_name]
        self.model = self.bundle["model"]
        self.calibrator = self.bundle.get("calibrator")
        self.features = self.bundle["features"]
        self.categorical_features = self.bundle.get("categorical_features", [])

        logger.info(f"Loaded Champion Model bundle '{self.bundle_name}' ({len(self.features)} features).")

    def reload(self) -> None:
        """
        Hot-reloads champion model bundle from disk into memory without restart.
        """
        self._load_model()
        logger.info(f"Hot-reloaded Champion Model bundle '{self.bundle_name}' successfully.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates calibrated purchase probabilities P(Y=1).
        """
        from sklearn.calibration import CalibratedClassifierCV
        features_subset = X[self.features].copy()
        for cat in getattr(self, "categorical_features", []):
            if cat in features_subset.columns:
                features_subset[cat] = features_subset[cat].astype("category")

        if isinstance(self.calibrator, CalibratedClassifierCV):
            p = self.calibrator.predict_proba(features_subset)[:, 1]
        else:
            p = self.model.predict_proba(features_subset)[:, 1]
            if self.calibrator is not None:
                p_clipped = np.clip(p, self.epsilon, 1.0 - self.epsilon)
                logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
                p = self.calibrator.predict_proba(logit_p)[:, 1]

        assert np.isfinite(p).all(), "Non-finite probabilities generated"
        assert ((p >= 0.0) & (p <= 1.0)).all(), "Probabilities outside [0, 1] range"
        return p

    def score_households(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores full household snapshot and assigns business deciles and action tiers.
        """
        df = snapshot_df.copy()
        p = self.predict_proba(df)
        df["purchase_probability"] = p

        # Rank: 1 = Highest Propensity
        df["rank"] = df["purchase_probability"].rank(ascending=False, method="first").astype(int)

        # Decile 10 = Top 10% highest propensity, Decile 1 = lowest
        df["decile"] = pd.qcut(
            df["purchase_probability"],
            q=10,
            labels=list(range(1, 11)),
        ).astype(int)

        # Urgency classification
        def get_urgency(row: pd.Series) -> str:
            prob = row["purchase_probability"]
            ratio = row.get("cadence_recency_ratio", 0.0)
            if prob >= 0.80 or (prob >= 0.60 and ratio >= 1.0):
                return "URGENT"
            elif prob >= 0.50:
                return "REGULAR"
            return "LOW"

        # Action strategy (Coupon Halo effect & cycle acceleration)
        def get_action(row: pd.Series) -> str:
            decile = row["decile"]
            if decile in (9, 10):
                return "VIP_COUPON_HALO_UPSELL"
            elif decile in (7, 8):
                return "CYCLE_ACCELERATION_NUDGE"
            elif decile in (5, 6):
                return "STANDARD_ENGAGEMENT"
            return "WINBACK_OR_HOLDOUT"

        df["urgency_tier"] = df.apply(get_urgency, axis=1)
        df["marketing_action"] = df.apply(get_action, axis=1)

        return df
