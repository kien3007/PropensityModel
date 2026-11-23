"""
Probability Calibration.

Calibrators map raw booster probabilities to calibrated probabilities. They are fitted
on the validation split (never on test) and chosen by cross-fitted Expected Calibration
Error: each candidate is fitted on k-1 folds of the validation predictions and scored on
the held-out fold, so a flexible calibrator (isotonic) cannot win by memorising.
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.training.evaluator import expected_calibration_error

logger = logging.getLogger(__name__)


class BaseCalibrator:
    method: str = "base"

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        raise NotImplementedError

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Returns calibrated P(Y=1) for an array of raw probabilities."""
        raise NotImplementedError


class IdentityCalibrator(BaseCalibrator):
    method = "none"

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        return np.asarray(raw_probs, dtype=float)


class PlattCalibrator(BaseCalibrator):
    """Platt scaling: logistic regression on logit(raw_probs)."""

    method = "platt"

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = epsilon
        self.calibrator: LogisticRegression | None = None

    def _logit(self, raw_probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw_probs, dtype=float), self.epsilon, 1.0 - self.epsilon)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(self._logit(raw_probs), np.asarray(y_true, dtype=int))
        self.calibrator = lr
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("PlattCalibrator is not fitted")
        return self.calibrator.predict_proba(self._logit(raw_probs))[:, 1]


class IsotonicCalibrator(BaseCalibrator):
    """Monotone, non-parametric calibration; clips outside the fitted range."""

    method = "isotonic"

    def __init__(self) -> None:
        self.calibrator: IsotonicRegression | None = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.asarray(raw_probs, dtype=float), np.asarray(y_true, dtype=float))
        self.calibrator = iso
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        return self.calibrator.predict(np.asarray(raw_probs, dtype=float))


CALIBRATORS: Dict[str, type[BaseCalibrator]] = {
    "none": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def make_calibrator(method: str) -> BaseCalibrator:
    try:
        return CALIBRATORS[method]()
    except KeyError:
        raise ValueError(f"Unknown calibration method '{method}'; choose from {sorted(CALIBRATORS)}") from None


def select_calibrator(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    methods: Sequence[str] = ("none", "platt", "isotonic"),
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[BaseCalibrator, Dict[str, Dict[str, float]]]:
    """
    Picks the calibration method with the lowest cross-fitted ECE, then refits it on all
    of `raw_probs`. Ties are broken in favour of the earlier (simpler) method.
    Returns (fitted calibrator, per-method report).
    """
    raw = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y_true, dtype=int)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    report: Dict[str, Dict[str, float]] = {}
    for method in methods:
        oof = np.empty_like(raw)
        for train_idx, hold_idx in folds.split(raw.reshape(-1, 1), y):
            cal = make_calibrator(method).fit(raw[train_idx], y[train_idx])
            oof[hold_idx] = cal.predict_proba(raw[hold_idx])
        report[method] = {"cv_ece": round(expected_calibration_error(y, oof), 6)}

    best = min(methods, key=lambda m: report[m]["cv_ece"])
    logger.info(f"Calibration selection (cross-fitted ECE): {report} -> '{best}'")
    return make_calibrator(best).fit(raw, y), report
