"""
Champion Model Predictor.

Loads the champion bundle, validates it against the feature contract, produces
calibrated purchase probabilities and assigns business deciles / action tiers.

Supported bundle layouts:
- schema v2 (src/training/model_registry.py): self-describing, written by this repo.
- schema v1 (legacy "bundles" dict exported from projectphase3): exactly one bundle,
  horizon read from `prediction_window_days`.
A bundle whose features do not match a known contract version is rejected instead of
being scored with whatever columns happen to exist.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.config.settings import settings
from src.features.contract import (
    FeatureContractError,
    apply_categorical_dtypes,
    infer_feature_set_version,
)
from src.training.calibrator import BaseCalibrator

logger = logging.getLogger(__name__)


class ModelBundleError(ValueError):
    """Raised when a model bundle is malformed or incompatible with the serving contract."""


def _normalise_bundle(package: Dict[str, Any]) -> Dict[str, Any]:
    if package.get("schema_version", 1) >= 2:
        spec = dict(package)
    else:
        bundles = package.get("bundles") or {}
        if len(bundles) != 1:
            raise ModelBundleError(
                f"Legacy bundle must contain exactly one model, found {len(bundles)}: {list(bundles)}"
            )
        name, inner = next(iter(bundles.items()))
        spec = {
            "schema_version": 1,
            "bundle_name": name,
            "model": inner["model"],
            "calibrator": inner.get("calibrator"),
            "features": inner["features"],
            "categorical_features": inner.get("categorical_features", []),
            "horizon_days": package.get("prediction_window_days"),
            "metrics": package.get("metrics", {}),
            "epsilon": package.get("epsilon", 1e-4),
        }
        try:
            spec["feature_set_version"] = infer_feature_set_version(spec["features"])
        except FeatureContractError as exc:
            raise ModelBundleError(str(exc)) from exc

    for key in ("model", "features", "feature_set_version"):
        if not spec.get(key):
            raise ModelBundleError(f"Model bundle is missing required field '{key}'")
    try:
        expected = infer_feature_set_version(spec["features"])
    except FeatureContractError as exc:
        raise ModelBundleError(str(exc)) from exc
    if expected != spec["feature_set_version"]:
        raise ModelBundleError(
            f"Bundle declares feature_set_version={spec['feature_set_version']} but its features match {expected}"
        )
    unknown_cats = set(spec.get("categorical_features", [])) - set(spec["features"])
    if unknown_cats:
        raise ModelBundleError(f"categorical_features not in features: {sorted(unknown_cats)}")
    if not spec.get("horizon_days"):
        raise ModelBundleError("Model bundle does not declare its prediction horizon (horizon_days)")
    return spec


class PropensityPredictor:
    def __init__(self, model_path: Optional[Path | str] = None, expected_horizon_days: Optional[int] = None) -> None:
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.expected_horizon_days = expected_horizon_days
        self.bundle: Dict[str, Any] = {}
        self.model = None
        self.calibrator = None
        self.features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_set_version: str = ""
        self.horizon_days: int = 0
        self.epsilon: float = 1e-4
        self.bundle_name: str = ""
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {self.model_path}")

        spec = _normalise_bundle(joblib.load(self.model_path))
        if self.expected_horizon_days is not None and spec["horizon_days"] != self.expected_horizon_days:
            raise ModelBundleError(
                f"Champion predicts a {spec['horizon_days']}-day horizon, "
                f"caller expects {self.expected_horizon_days} days"
            )

        self.bundle = spec
        self.bundle_name = spec["bundle_name"]
        self.model = spec["model"]
        self.calibrator = spec.get("calibrator")
        self.features = list(spec["features"])
        self.categorical_features = list(spec.get("categorical_features", []))
        self.feature_set_version = spec["feature_set_version"]
        self.horizon_days = int(spec["horizon_days"])
        self.epsilon = spec.get("epsilon", 1e-4)
        logger.info(
            f"Loaded champion '{self.bundle_name}' (schema v{spec['schema_version']}, "
            f"{self.feature_set_version}, {len(self.features)} features, horizon {self.horizon_days}d)."
        )

    def reload(self) -> None:
        """Hot-reloads the champion bundle from disk without restarting the service."""
        self._load_model()

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise FeatureContractError(f"Scoring frame is missing {len(missing)} model features: {missing}")
        return apply_categorical_dtypes(X[self.features], self.categorical_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(Y=1) per row."""
        features = self._prepare(X)
        cal = self.calibrator
        if isinstance(cal, CalibratedClassifierCV):
            p = cal.predict_proba(features)[:, 1]
        else:
            raw = self.model.predict_proba(features)[:, 1]
            if cal is None:
                p = raw
            elif isinstance(cal, BaseCalibrator):
                p = cal.predict_proba(raw)
            elif isinstance(cal, LogisticRegression):  # legacy Platt on logit(raw)
                clipped = np.clip(raw, self.epsilon, 1.0 - self.epsilon)
                p = cal.predict_proba(np.log(clipped / (1.0 - clipped)).reshape(-1, 1))[:, 1]
            else:
                raise ModelBundleError(f"Unsupported calibrator type {type(cal).__name__}")

        p = np.asarray(p, dtype=float)
        if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
            raise ValueError("Model produced probabilities outside [0, 1]")
        return p

    def score_households(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """Scores a household snapshot and assigns business deciles and action tiers."""
        df = snapshot_df.copy()
        df["purchase_probability"] = self.predict_proba(df)

        # Rank: 1 = highest propensity. Deciles are rank-based so that ties (isotonic
        # calibration produces plateaus) cannot collapse quantile bin edges.
        df["rank"] = df["purchase_probability"].rank(ascending=False, method="first").astype(int)
        df["decile"] = pd.qcut(-df["rank"], q=10, labels=list(range(1, 11))).astype(int)

        prob = df["purchase_probability"]
        overdue = df["urgency_ratio"] if "urgency_ratio" in df.columns else pd.Series(0.0, index=df.index)
        df["urgency_tier"] = np.select(
            [(prob >= 0.80) | ((prob >= 0.60) & (overdue >= 1.0)), prob >= 0.50],
            ["URGENT", "REGULAR"],
            default="LOW",
        )
        df["marketing_action"] = np.select(
            [df["decile"] >= 9, df["decile"] >= 7, df["decile"] >= 5],
            ["VIP_COUPON_HALO_UPSELL", "CYCLE_ACCELERATION_NUDGE", "STANDARD_ENGAGEMENT"],
            default="WINBACK_OR_HOLDOUT",
        )
        return df
