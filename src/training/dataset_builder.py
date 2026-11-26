"""
Dataset Builder for training, validation and out-of-time test splits.

Each row is one eligible (household, cutoff) pair: features computed strictly from
DAY <= cutoff by the shared `MultiDomainFeatureBuilder`; the label and the eligibility
rule come from the configured `TargetSpec` (src/features/targets.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Sequence

import pandas as pd

from src.config.model_config import TrainingConfig
from src.features.contract import CATEGORICAL_FEATURES, MODEL_FEATURES, apply_categorical_dtypes, features_for
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.features.targets import PURCHASE, TargetSpec, get_target

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    name: str
    frame: pd.DataFrame  # household_key, cutoff_day, <features>, target
    features: List[str] = field(default_factory=lambda: list(MODEL_FEATURES))

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.features]

    @property
    def y(self) -> pd.Series:
        return self.frame["target"]

    @property
    def positive_rate(self) -> float:
        return float(self.frame["target"].mean())


class DatasetBuilder:
    """Builds leakage-free multi-cutoff datasets from one `MultiDomainFeatureBuilder`."""

    def __init__(self, features: MultiDomainFeatureBuilder) -> None:
        self.features = features

    def build_snapshot(self, cutoff: int, horizon_days: int, target: TargetSpec = PURCHASE) -> pd.DataFrame:
        last_day = self.features.last_observed_day()
        if cutoff + horizon_days > last_day:
            raise ValueError(
                f"Cutoff {cutoff} + horizon {horizon_days} exceeds last observed DAY {last_day}: "
                "labels would be incomplete"
            )
        feats = self.features.features_at(cutoff)
        feats = feats[target.eligible(feats)]
        labels = target.labels(self.features, cutoff, horizon_days)
        snapshot = feats.merge(labels, on=["household_key", "cutoff_day"], how="inner", validate="one_to_one")
        logger.info(
            f"Snapshot DAY={cutoff} ({target.name}, horizon={horizon_days}d): {len(snapshot):,} eligible households, "
            f"positive rate {snapshot['target'].mean():.4f}"
        )
        return snapshot

    def build(self, cutoffs: Sequence[int], horizon_days: int, target: TargetSpec = PURCHASE) -> pd.DataFrame:
        frames = [self.build_snapshot(c, horizon_days, target) for c in sorted(cutoffs)]
        # Re-apply categorical dtypes after concat: per-snapshot category sets differ and
        # pd.concat would otherwise degrade them to object dtype.
        return apply_categorical_dtypes(pd.concat(frames, ignore_index=True), CATEGORICAL_FEATURES)

    def build_splits(self, config: TrainingConfig) -> Dict[str, TemporalSplit]:
        version = getattr(self.features, "feature_set_version", config.feature_set_version)
        if version != config.feature_set_version:
            raise ValueError(f"Feature source computes {version}, config expects {config.feature_set_version}")
        features = features_for(config.feature_set_version)
        target = get_target(config.target)
        splits = {
            name: TemporalSplit(name, self.build(cutoffs, config.horizon_days, target), features)
            for name, cutoffs in (("fit", config.fit_cutoffs), ("val", config.val_cutoffs), ("test", config.test_cutoffs))
        }
        for split in splits.values():
            logger.info(f"Split '{split.name}': {len(split.frame):,} rows, positive rate {split.positive_rate:.4f}")
        return splits
