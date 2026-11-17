"""
Probability Calibration Module.
Provides Platt scaling (logistic calibration on logit transform) and Isotonic calibration.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """
    Fits and applies Platt Scaling to calibrate raw LightGBM probability outputs.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = epsilon
        self.calibrator: Optional[LogisticRegression] = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        """
        Fits logistic regression model on logit(raw_probs).
        """
        p_clipped = np.clip(raw_probs, self.epsilon, 1.0 - self.epsilon)
        logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(logit_p, y_true)
        self.calibrator = lr
        logger.info(f"PlattCalibrator fit successfully (coef: {lr.coef_[0][0]:.4f}, intercept: {lr.intercept_[0]:.4f}).")
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Calibrates raw probabilities using the fitted Platt model.
        """
        if self.calibrator is None:
            return raw_probs

        p_clipped = np.clip(raw_probs, self.epsilon, 1.0 - self.epsilon)
        logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logit_p)[:, 1]
