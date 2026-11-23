"""
Single source of truth for the model definition: prediction target, temporal splits,
LightGBM hyper-parameters, calibration search and promotion-gate thresholds.

Every pipeline (training, batch scoring, DAGs, scheduler, reproduction script) reads
from here so the champion's definition cannot silently drift between components.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class GateConfig:
    """Absolute quality thresholds a candidate must meet on the out-of-time test set."""

    min_auc: float = 0.75
    max_brier: float = 0.20
    min_lift_10: float = 1.50


@dataclass(frozen=True)
class TrainingConfig:
    # Feature contract version the model is trained against (see src/features/contract.py).
    feature_set_version: str = "multidomain_v1"

    # Target: household makes >= 1 valid basket in (cutoff, cutoff + horizon_days].
    horizon_days: int = 14
    # Observation window used by the point-in-time feature extractor.
    lookback_days: int = 180

    # Temporal splits (Dunnhumby DAY 1..711). Must be separated by >= horizon_days so
    # label windows of one split never overlap the feature window of the next.
    fit_cutoffs: Tuple[int, ...] = (200, 230, 260, 290, 320, 350, 380, 410, 440, 470, 500)
    val_cutoffs: Tuple[int, ...] = (530, 560, 590)
    test_cutoffs: Tuple[int, ...] = (650, 680)

    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            # LightGBM ignores `subsample` unless bagging frequency > 0.
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )
    early_stopping_rounds: int = 50

    # Calibration candidates compared by cross-fitted ECE on the validation predictions.
    calibration_methods: Tuple[str, ...] = ("none", "platt", "isotonic")
    calibration_cv_folds: int = 5

    gate: GateConfig = field(default_factory=GateConfig)

    # Acceptable |ROC-AUC difference| between a retrained model and the active champion
    # on the same test cutoffs for the champion to count as reproduced.
    reproduction_auc_tolerance: float = 0.01

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.horizon_days <= 0 or self.lookback_days <= 0:
            raise ValueError("horizon_days and lookback_days must be positive")
        splits = {"fit": self.fit_cutoffs, "val": self.val_cutoffs, "test": self.test_cutoffs}
        for name, cutoffs in splits.items():
            if not cutoffs:
                raise ValueError(f"{name}_cutoffs must not be empty")
            if list(cutoffs) != sorted(set(cutoffs)):
                raise ValueError(f"{name}_cutoffs must be strictly increasing: {cutoffs}")
        ordered = [("fit", self.fit_cutoffs), ("val", self.val_cutoffs), ("test", self.test_cutoffs)]
        for (prev_name, prev), (next_name, nxt) in zip(ordered, ordered[1:]):
            if max(prev) + self.horizon_days > min(nxt):
                raise ValueError(
                    f"Temporal leakage: last {prev_name} cutoff {max(prev)} + horizon {self.horizon_days} "
                    f"overlaps first {next_name} cutoff {min(nxt)}"
                )
        unknown = set(self.calibration_methods) - {"none", "platt", "isotonic"}
        if unknown:
            raise ValueError(f"Unknown calibration methods: {sorted(unknown)}")

    def with_overrides(self, **changes: Any) -> "TrainingConfig":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_set_version": self.feature_set_version,
            "horizon_days": self.horizon_days,
            "lookback_days": self.lookback_days,
            "fit_cutoffs": list(self.fit_cutoffs),
            "val_cutoffs": list(self.val_cutoffs),
            "test_cutoffs": list(self.test_cutoffs),
            "lgbm_params": dict(self.lgbm_params),
            "early_stopping_rounds": self.early_stopping_rounds,
            "calibration_methods": list(self.calibration_methods),
            "calibration_cv_folds": self.calibration_cv_folds,
            "gate": vars(self.gate).copy(),
        }


DEFAULT_TRAINING_CONFIG = TrainingConfig()
