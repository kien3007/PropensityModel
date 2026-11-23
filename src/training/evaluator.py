"""
Model Evaluation Metrics: discrimination (ROC-AUC, PR-AUC, lift), calibration
(Brier, log loss, ECE) and the business decile table.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Sample-weighted mean |observed rate - mean predicted| over equal-width probability bins."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    bins = np.minimum((y_p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_t[mask].mean() - y_p[mask].mean())
    return float(ece)


class ModelEvaluator:
    """Evaluates probability predictions against ground-truth binary targets."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        epsilon: float = 1e-6,
    ) -> Dict[str, Any]:
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.clip(np.asarray(y_prob, dtype=float), epsilon, 1.0 - epsilon)

        base_rate = float(y_t.mean())
        k = max(1, int(len(y_t) * 0.10))
        top_indices = np.argsort(-y_p, kind="stable")[:k]
        top_precision = float(y_t[top_indices].mean())
        lift_10 = top_precision / base_rate if base_rate > 0 else 1.0

        metrics: Dict[str, Any] = {
            "roc_auc": round(float(roc_auc_score(y_t, y_p)), 6),
            "pr_auc": round(float(average_precision_score(y_t, y_p)), 6),
            "brier_score": round(float(brier_score_loss(y_t, y_p)), 6),
            "log_loss": round(float(log_loss(y_t, y_p)), 6),
            "ece": round(expected_calibration_error(y_t, y_p), 6),
            "base_rate": round(base_rate, 4),
            "top_10_precision": round(top_precision, 4),
            "lift_at_10pct": round(lift_10, 4),
            # Upper bound of lift@10% given the base rate (all of the top decile buys).
            "max_attainable_lift_10": round(1.0 / base_rate, 4) if base_rate > 0 else 1.0,
            "sample_size": int(len(y_t)),
        }
        metrics["top_decile_lift"] = metrics["lift_at_10pct"]
        logger.info(
            f"Evaluation: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
            f"Brier={metrics['brier_score']:.4f}, ECE={metrics['ece']:.4f}, "
            f"Lift@10%={metrics['lift_at_10pct']:.3f}x (max {metrics['max_attainable_lift_10']:.3f}x)"
        )
        return metrics
