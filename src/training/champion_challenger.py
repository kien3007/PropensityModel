"""
Champion / Challenger comparison.

The challenger replaces the champion only if it is not worse on the *same* out-of-time
test rows: the lower bound of the 95% household-clustered bootstrap CI of
AUC(challenger) - AUC(champion) must be >= -non_inferiority_margin_auc.

If the champion predicts a different target (horizon / feature set), the two models are
not comparable and the challenger is blocked: changing the target is a deliberate
decision that needs `force_promote`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config.model_config import TrainingConfig
from src.features.contract import FeatureContractError
from src.models.predictor import ModelBundleError, PropensityPredictor
from src.training.evaluator import ModelEvaluator
from src.training.statistics import paired_bootstrap_delta

logger = logging.getLogger(__name__)


def compare_with_champion(
    challenger_probs: np.ndarray,
    test_frame: pd.DataFrame,
    champion_path: Path,
    config: TrainingConfig,
) -> Dict[str, Any]:
    gate = config.gate
    if not Path(champion_path).exists():
        logger.info("No active champion: challenger comparison skipped.")
        return {"status": "no_champion", "passed": True}

    try:
        champion = PropensityPredictor(model_path=champion_path)
    except (ModelBundleError, FeatureContractError) as exc:
        return {"status": "incomparable", "passed": False, "reason": f"champion bundle unusable: {exc}"}

    champion_def = (champion.target, champion.feature_set_version, champion.horizon_days)
    challenger_def = (config.target, config.feature_set_version, config.horizon_days)
    if champion_def != challenger_def:
        reason = (
            f"target changed: champion {champion_def} vs challenger {challenger_def} "
            "(target, feature set, horizon); promote with force_promote"
        )
        logger.warning(f"Champion comparison not possible: {reason}")
        return {"status": "incomparable", "passed": False, "reason": reason}

    y = test_frame["target"].to_numpy()
    champion_probs = champion.predict_proba(test_frame)
    boot = paired_bootstrap_delta(
        y, challenger_probs, champion_probs,
        groups=test_frame["household_key"].to_numpy(),
        n_samples=gate.bootstrap_samples, seed=gate.bootstrap_seed,
    )
    passed = boot["ci_low"] >= -gate.non_inferiority_margin_auc

    # A champion trained on data overlapping the test window has an unfair advantage;
    # the comparison is then conservative (biased against the challenger) - flag it.
    trained_until = max(list(champion.bundle.get("fit_cutoffs") or [0]) + list(champion.bundle.get("val_cutoffs") or [0]))
    overlap = trained_until + champion.horizon_days > min(config.test_cutoffs)

    result = {
        "status": "compared",
        "passed": bool(passed),
        "champion_bundle": champion.bundle_name,
        "champion_metrics": ModelEvaluator.evaluate(y, champion_probs),
        "auc_delta": round(boot["delta"], 6),
        "auc_delta_ci95": [round(boot["ci_low"], 6), round(boot["ci_high"], 6)],
        "p_challenger_not_better": round(boot["p_a_not_better"], 4),
        "non_inferiority_margin_auc": gate.non_inferiority_margin_auc,
        "bootstrap_samples": boot["n_samples"],
        "champion_training_overlaps_test": bool(overlap),
    }
    log = logger.info if passed else logger.warning
    log(
        f"Challenger vs champion: dAUC={boot['delta']:+.4f} "
        f"(95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}], margin {-gate.non_inferiority_margin_auc:+.4f}) "
        f"-> {'non-inferior' if passed else 'WORSE than champion'}"
    )
    return result
