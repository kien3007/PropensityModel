"""
Paired, household-clustered bootstrap for comparing two models on the same rows.

The out-of-time test set stacks several cutoffs, so each household appears several
times; resampling households (not rows) keeps those correlated rows together and gives
honest confidence intervals.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def paired_bootstrap_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    groups: Optional[np.ndarray] = None,
    metric: MetricFn = roc_auc_score,
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Distribution of metric(a) - metric(b) under a (cluster) bootstrap of the rows."""
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_true, prob_a and prob_b must have the same length")

    if groups is None:
        groups = np.arange(len(y))
    _, group_idx = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(group_idx == g) for g in range(group_idx.max() + 1)]
    n_groups = len(members)

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_samples):
        picked = rng.integers(0, n_groups, n_groups)
        idx = np.concatenate([members[g] for g in picked])
        if y[idx].min() == y[idx].max():  # degenerate resample (single class)
            continue
        deltas.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    if not deltas:
        raise ValueError("Every bootstrap resample was single-class; cannot compare models")

    d = np.asarray(deltas)
    return {
        "delta": float(metric(y, a) - metric(y, b)),
        "ci_low": float(np.quantile(d, alpha / 2)),
        "ci_high": float(np.quantile(d, 1 - alpha / 2)),
        "p_a_not_better": float((d <= 0).mean()),
        "n_samples": int(len(d)),
    }
