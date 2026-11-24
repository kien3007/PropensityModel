"""
Feature quality gate: declarative checks a snapshot must pass before it is stored or scored.

The contract (validate_feature_frame) only guarantees columns exist and numerics are finite.
These checks catch values that are finite but wrong:
  - bounds        every numeric feature has a valid range (and integer-ness where it is a count)
  - categories    categorical values must be ones the model was trained on
  - consistency   rolling-window counts and spends cannot shrink as the window grows
  - coverage      one row per household, and no sudden loss of households vs the previous snapshot
  - degeneration  the share of zeros in a feature cannot jump vs the previous snapshot
                  (typical symptom of a broken upstream join that fills everything with 0)

Thresholds were calibrated on the 17 real Dunnhumby snapshots (DAY 200..711): no real snapshot
violates them, the largest real zero-share jump between consecutive snapshots is 0.04
(excluding the campaign flag, which legitimately flips when campaigns start or end).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.contract import CATEGORICAL_FEATURES, CATEGORICAL_FILL_VALUE, NUMERIC_FEATURES

DEFAULT_LOOKBACK_DAYS = 180
MIN_COVERAGE_RATIO = 0.90       # a snapshot may not have < 90% of the previous snapshot's households
MAX_ZERO_SHARE_JUMP = 0.25      # share of zeros in a feature may not move by more than 25 points
MAX_REPORTED_VIOLATIONS = 20


@dataclass(frozen=True)
class Bound:
    low: Optional[float] = 0.0
    high: Optional[float] = None
    integer: bool = False


def numeric_bounds(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, Bound]:
    """Valid range of every numeric feature (follows the formulas in pit_extractor)."""
    never_seen = lookback_days + 1  # value used for households without history in the lookback
    bounds = {
        "days_since_last_basket": Bound(0, never_seen),
        "days_since_last_perishable": Bound(0, never_seen),
        "median_inter_trip": Bound(0),  # inter-trip gaps use the full history, not only the lookback
        "std_inter_trip": Bound(0),
        "urgency_ratio": Bound(0),
        "frequency_momentum_30_vs_90": Bound(0, 3.0),  # (c30/30) / (c90/90) with c30 <= c90
        "spend_30d": Bound(0),
        "spend_90d": Bound(0),
        "spend_180d": Bound(0),
        "aov_90d": Bound(0),
        "avg_items_per_basket_90d": Bound(0),
        "perishable_basket_ratio_90d": Bound(0, 1),
        "is_targeted_by_active_campaign": Bound(0, 1, integer=True),
        "coupon_redemption_count_180d": Bound(0, integer=True),
        "primary_store_id": Bound(0, integer=True),
        "store_loyalty_index": Bound(0, 1),
    }
    for window in (14, 30, 60, 90, 180):
        bounds[f"baskets_count_{window}d"] = Bound(0, integer=True)
    missing = set(NUMERIC_FEATURES) - set(bounds)
    if missing:  # a new numeric feature must declare its range here
        raise RuntimeError(f"No quality bound declared for numeric features: {sorted(missing)}")
    return bounds


# Values seen in the Dunnhumby demographics (plus the contract's fill value). A new value means
# upstream changed its coding; the model has never seen it, so it must be reviewed first.
ALLOWED_CATEGORIES: Dict[str, frozenset] = {
    col: frozenset(values) | {CATEGORICAL_FILL_VALUE}
    for col, values in {
        "classification_1": [f"Age Group{i}" for i in range(1, 7)],
        "classification_2": ["X", "Y", "Z"],
        "classification_3": [f"Level{i}" for i in range(1, 13)],
        "HOMEOWNER_DESC": ["Homeowner", "Probable Owner", "Probable Renter", "Renter", "Unknown"],
        "classification_4": ["1", "2", "3", "4", "5+"],
        "classification_5": [f"Group{i}" for i in range(1, 7)],
        "KID_CATEGORY_DESC": ["1", "2", "3+", "None/Unknown"],
    }.items()
}
assert set(ALLOWED_CATEGORIES) == set(CATEGORICAL_FEATURES)

# Each chain must be non-decreasing row by row (a longer window contains the shorter one).
MONOTONE_CHAINS: List[List[str]] = [
    [f"baskets_count_{w}d" for w in (14, 30, 60, 90, 180)],
    ["spend_30d", "spend_90d", "spend_180d"],
]

# Features whose zero share legitimately swings between snapshots.
ZERO_SHARE_EXEMPT = frozenset({"is_targeted_by_active_campaign"})


class FeatureQualityError(ValueError):
    """Raised when a feature snapshot fails the quality gate. `.violations` lists every failure."""

    def __init__(self, violations: List[str], cutoff_day: Optional[int] = None) -> None:
        self.violations = violations
        where = f" for DAY {cutoff_day}" if cutoff_day is not None else ""
        shown = violations[:MAX_REPORTED_VIOLATIONS]
        more = len(violations) - len(shown)
        super().__init__(
            f"Feature snapshot{where} failed {len(violations)} quality check(s): " + "; ".join(shown)
            + (f"; ... and {more} more" if more > 0 else "")
        )


def _zero_share(df: pd.DataFrame, col: str) -> float:
    return float((df[col].to_numpy(dtype=np.float64) == 0).mean()) if len(df) else 0.0


def find_quality_violations(
    features: pd.DataFrame,
    previous: Optional[pd.DataFrame] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_coverage_ratio: float = MIN_COVERAGE_RATIO,
    max_zero_share_jump: float = MAX_ZERO_SHARE_JUMP,
) -> List[str]:
    """Returns human-readable violations (empty list = the snapshot passes)."""
    violations: List[str] = []
    if features.empty:
        return ["snapshot is empty"]

    # coverage
    dupes = int(features["household_key"].duplicated().sum())
    if dupes:
        violations.append(f"{dupes} duplicated household_key rows")
    if previous is not None and len(previous):
        ratio = len(features) / len(previous)
        if ratio < min_coverage_ratio:
            violations.append(
                f"coverage dropped to {len(features)} households from {len(previous)} "
                f"({ratio:.0%} < {min_coverage_ratio:.0%})"
            )

    # bounds
    for col, bound in numeric_bounds(lookback_days).items():
        values = features[col].to_numpy(dtype=np.float64)
        if bound.low is not None and (n := int((values < bound.low).sum())):
            violations.append(f"{col}: {n} values below {bound.low:g} (min {values.min():g})")
        if bound.high is not None and (n := int((values > bound.high + 1e-9).sum())):
            violations.append(f"{col}: {n} values above {bound.high:g} (max {values.max():g})")
        if bound.integer and (n := int((values != np.round(values)).sum())):
            violations.append(f"{col}: {n} non-integer values")

    # categories
    for col, allowed in ALLOWED_CATEGORIES.items():
        unknown = sorted(set(features[col].astype(str).unique()) - allowed)
        if unknown:
            violations.append(f"{col}: unseen categories {unknown[:5]}")

    # consistency between windows
    for chain in MONOTONE_CHAINS:
        for shorter, longer in zip(chain, chain[1:]):
            n = int((features[shorter].to_numpy(dtype=np.float64)
                     > features[longer].to_numpy(dtype=np.float64) + 1e-6).sum())
            if n:
                violations.append(f"{shorter} > {longer} for {n} households")

    # degeneration vs the previous snapshot
    if previous is not None and len(previous):
        for col in NUMERIC_FEATURES:
            if col in ZERO_SHARE_EXEMPT:
                continue
            now, before = _zero_share(features, col), _zero_share(previous, col)
            if abs(now - before) > max_zero_share_jump:
                violations.append(f"{col}: share of zeros moved from {before:.0%} to {now:.0%}")
    return violations


def check_feature_quality(
    features: pd.DataFrame,
    previous: Optional[pd.DataFrame] = None,
    cutoff_day: Optional[int] = None,
    **kwargs,
) -> None:
    """Raises FeatureQualityError if the snapshot fails any check."""
    violations = find_quality_violations(features, previous, **kwargs)
    if violations:
        raise FeatureQualityError(violations, cutoff_day)
