"""
Prediction targets.

A target defines (a) which households are eligible to be scored at a cutoff and (b) the
binary label over (cutoff, cutoff + horizon]. Both use only the shared feature builder, so
eligibility at training time and at batch-scoring time is computed identically.

- purchase: every household; 1 = at least one valid basket in the horizon.
  At a 14-day horizon the base rate is ~0.70 and is driven almost entirely by shopping
  frequency (20+ baskets in 90d -> 97% buy), so ranking value is limited.
- lapse: regular shoppers only (>= 1 basket in the last 30 days AND >= 4 in the last 90);
  1 = no valid basket in the horizon. At 21 days the base rate is ~0.11 over ~1,850
  eligible households: a retention signal where ranking matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import pandas as pd

if TYPE_CHECKING:
    from src.features.pit_extractor import MultiDomainFeatureBuilder


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str
    min_baskets_30d: int = 0
    min_baskets_90d: int = 0
    invert_purchase: bool = False

    def eligible(self, features: pd.DataFrame) -> pd.Series:
        """Boolean mask of households scored / trained on at this cutoff."""
        return (features["baskets_count_30d"] >= self.min_baskets_30d) & (
            features["baskets_count_90d"] >= self.min_baskets_90d
        )

    def labels(self, builder: "MultiDomainFeatureBuilder", cutoff: int, horizon_days: int) -> pd.DataFrame:
        labels = builder.labels_at(cutoff, horizon_days)
        if self.invert_purchase:
            labels["target"] = 1 - labels["target"]
        return labels


PURCHASE = TargetSpec(
    name="purchase",
    description="At least one valid basket in (cutoff, cutoff + horizon]",
)
LAPSE = TargetSpec(
    name="lapse",
    description="Regular shopper (>=1 basket in 30d and >=4 in 90d) with no valid basket in (cutoff, cutoff + horizon]",
    min_baskets_30d=1,
    min_baskets_90d=4,
    invert_purchase=True,
)

TARGETS: Dict[str, TargetSpec] = {t.name: t for t in (PURCHASE, LAPSE)}


def get_target(name: str) -> TargetSpec:
    try:
        return TARGETS[name]
    except KeyError:
        raise ValueError(f"Unknown target '{name}'; choose from {sorted(TARGETS)}") from None
