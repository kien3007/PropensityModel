"""
Feature Contract: the one definition of what a model consumes.

Training, batch scoring, the feature store and the predictor all import the feature
lists and dtype rules from here, so a feature added or renamed in one place cannot
silently diverge from another (train/serve skew).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

FEATURE_SET_VERSION = "multidomain_v1"

# Champion 28 multi-domain features (order matters: it is the model's column order).
MODEL_FEATURES: List[str] = [
    "days_since_last_basket",
    "days_since_last_perishable",
    "median_inter_trip",
    "std_inter_trip",
    "urgency_ratio",
    "baskets_count_14d",
    "baskets_count_30d",
    "baskets_count_60d",
    "baskets_count_90d",
    "baskets_count_180d",
    "frequency_momentum_30_vs_90",
    "spend_30d",
    "spend_90d",
    "spend_180d",
    "aov_90d",
    "avg_items_per_basket_90d",
    "perishable_basket_ratio_90d",
    "is_targeted_by_active_campaign",
    "coupon_redemption_count_180d",
    "primary_store_id",
    "store_loyalty_index",
    "classification_1",
    "classification_2",
    "classification_3",
    "HOMEOWNER_DESC",
    "classification_4",
    "classification_5",
    "KID_CATEGORY_DESC",
]

CATEGORICAL_FEATURES: List[str] = [
    "classification_1",
    "classification_2",
    "classification_3",
    "HOMEOWNER_DESC",
    "classification_4",
    "classification_5",
    "KID_CATEGORY_DESC",
]

NUMERIC_FEATURES: List[str] = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]

CATEGORICAL_FILL_VALUE = "UNKNOWN"

# Legacy 23 RFM/cadence features of the retired "18 + chu kỳ" bundle. Kept only so the
# legacy extractor (and its tests) still work; no supported model uses them.
LEGACY_FEATURES_23: List[str] = [
    "recency_days",
    "observed_tenure_days",
    "baskets_7d",
    "sales_7d",
    "line_items_7d",
    "active_days_7d",
    "baskets_30d",
    "sales_30d",
    "line_items_30d",
    "active_days_30d",
    "baskets_90d",
    "sales_90d",
    "line_items_90d",
    "active_days_90d",
    "baskets_180d",
    "sales_180d",
    "line_items_180d",
    "active_days_180d",
    "cadence_median_gap_180d",
    "cadence_gap_std_180d",
    "cadence_recency_ratio",
    "cadence_has_gap",
    "cadence_has_two_gaps",
]

FEATURE_SETS: Dict[str, List[str]] = {FEATURE_SET_VERSION: MODEL_FEATURES}


class FeatureContractError(ValueError):
    """Raised when a frame or model bundle does not satisfy the feature contract."""


def infer_feature_set_version(features: Sequence[str]) -> str:
    """Maps a model's feature list onto a known contract version, or raises."""
    for version, contract_features in FEATURE_SETS.items():
        if list(features) == contract_features:
            return version
    raise FeatureContractError(
        f"Feature list ({len(features)} features) does not match any supported feature set "
        f"{sorted(FEATURE_SETS)}; first features: {list(features)[:5]}"
    )


def apply_categorical_dtypes(df: pd.DataFrame, categorical: Sequence[str] = CATEGORICAL_FEATURES) -> pd.DataFrame:
    """
    Normalises categorical columns to pandas 'category' over string values.

    Values are cast to str first so that CSV- and Parquet-loaded data (which may type a
    column such as classification_4 as int or str) map onto identical categories.
    Apply after concatenating snapshots: per-snapshot category sets differ, and
    concatenating mismatched categoricals degrades them to object dtype.
    """
    out = df.copy()
    for col in categorical:
        if col not in out.columns:
            out[col] = CATEGORICAL_FILL_VALUE
        values = out[col].astype(object).where(out[col].notna(), CATEGORICAL_FILL_VALUE)
        out[col] = values.astype(str).astype("category")
    return out


def validate_feature_frame(df: pd.DataFrame, features: Sequence[str] = MODEL_FEATURES) -> None:
    """Fails loudly on missing columns or non-finite numeric values."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise FeatureContractError(f"Feature frame is missing {len(missing)} contract columns: {missing}")
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    values = df[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = [c for c in numeric if not np.isfinite(df[c].to_numpy(dtype=np.float64)).all()]
        raise FeatureContractError(f"Non-finite values in numeric features: {bad}")
