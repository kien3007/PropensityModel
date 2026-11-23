"""
Dataset Builder for training, validation and out-of-time test splits.

Each row is one (household, cutoff) pair: features computed strictly from DAY <= cutoff
by the shared `MultiDomainFeatureBuilder`, target = purchase in (cutoff, cutoff + horizon].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd

from src.config.model_config import TrainingConfig
from src.features.contract import CATEGORICAL_FEATURES, MODEL_FEATURES, apply_categorical_dtypes
from src.features.pit_extractor import MultiDomainFeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    name: str
    frame: pd.DataFrame  # household_key, cutoff_day, <features>, target

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[MODEL_FEATURES]

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

    def build_snapshot(self, cutoff: int, horizon_days: int) -> pd.DataFrame:
        last_day = self.features.last_observed_day()
        if cutoff + horizon_days > last_day:
            raise ValueError(
                f"Cutoff {cutoff} + horizon {horizon_days} exceeds last observed DAY {last_day}: "
                "labels would be incomplete"
            )
        feats = self.features.features_at(cutoff)
        labels = self.features.labels_at(cutoff, horizon_days)
        snapshot = feats.merge(labels, on=["household_key", "cutoff_day"], how="inner", validate="one_to_one")
        logger.info(
            f"Snapshot DAY={cutoff} (horizon={horizon_days}d): {len(snapshot):,} households, "
            f"positive rate {snapshot['target'].mean():.4f}"
        )
        return snapshot

    def build(self, cutoffs: Sequence[int], horizon_days: int) -> pd.DataFrame:
        frames = [self.build_snapshot(c, horizon_days) for c in sorted(cutoffs)]
        # Re-apply categorical dtypes after concat: per-snapshot category sets differ and
        # pd.concat would otherwise degrade them to object dtype.
        return apply_categorical_dtypes(pd.concat(frames, ignore_index=True), CATEGORICAL_FEATURES)

    def build_splits(self, config: TrainingConfig) -> Dict[str, TemporalSplit]:
        splits = {
            "fit": TemporalSplit("fit", self.build(config.fit_cutoffs, config.horizon_days)),
            "val": TemporalSplit("val", self.build(config.val_cutoffs, config.horizon_days)),
            "test": TemporalSplit("test", self.build(config.test_cutoffs, config.horizon_days)),
        }
        for split in splits.values():
            logger.info(f"Split '{split.name}': {len(split.frame):,} rows, positive rate {split.positive_rate:.4f}")
        return splits
