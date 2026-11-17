"""
Model Evaluation and Metric Calculation Module.
Calculates calibration and discrimination metrics: Brier score, Log loss, ROC-AUC, Lift@10%.
"""
from __future__ import annotations

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates probability predictions against ground-truth binary targets.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        epsilon: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Calculates production-grade evaluation metrics.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_prob, dtype=float)
        y_p = np.clip(y_p, epsilon, 1.0 - epsilon)

        brier = float(brier_score_loss(y_t, y_p))
        ll = float(log_loss(y_t, y_p))
        auc = float(roc_auc_score(y_t, y_p))

        # Top 10% Decile Lift Calculation
        k = max(1, int(len(y_t) * 0.10))
        top_indices = np.argsort(y_p)[::-1][:k]
        top_precision = float(y_t[top_indices].mean())
        base_rate = float(y_t.mean())
        lift_10 = float(top_precision / base_rate) if base_rate > 0 else 1.0

        metrics = {
            "brier_score": round(brier, 6),
            "log_loss": round(ll, 6),
            "roc_auc": round(auc, 6),
            "base_rate": round(base_rate, 4),
            "top_10_precision": round(top_precision, 4),
            "lift_at_10pct": round(lift_10, 4),
            "sample_size": len(y_t),
        }
        logger.info(
            f"Evaluation Metrics: ROC-AUC={metrics['roc_auc']:.4f}, "
            f"Brier={metrics['brier_score']:.4f}, Lift@10%={metrics['lift_at_10pct']:.2f}x"
        )
        return metrics
