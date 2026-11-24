"""
Feast object definitions, generated from the feature contract so the store schema cannot
drift from what the model consumes.

Dunnhumby has no calendar dates, so DAY d is mapped onto a synthetic timeline
(FEATURE_EPOCH + d days, UTC). A snapshot computed at cutoff t0 carries event_timestamp =
day_to_timestamp(t0); Feast point-in-time joins then return, for an entity at time T, the
latest snapshot with event_timestamp <= T and within the TTL.
"""
from __future__ import annotations

import datetime
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, String
from feast.value_type import ValueType

from src.features.contract import CATEGORICAL_FEATURES, FEATURE_SET_VERSION, MODEL_FEATURES

FEATURE_EPOCH = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
FEATURE_VIEW_NAME = f"household_features_{FEATURE_SET_VERSION}"
PROJECT_NAME = "propensity"


def day_to_timestamp(day: int) -> datetime.datetime:
    return FEATURE_EPOCH + datetime.timedelta(days=int(day))


def timestamp_to_day(ts) -> int:
    return int(round((ts - FEATURE_EPOCH).total_seconds() / 86400))


household = Entity(
    name="household",
    join_keys=["household_key"],
    value_type=ValueType.INT64,
    description="Dunnhumby household",
)


def build_feature_view(offline_path: str, ttl_days: int) -> FeatureView:
    schema = [
        Field(name=f, dtype=String if f in CATEGORICAL_FEATURES else Float64)
        for f in MODEL_FEATURES
    ]
    source = FileSource(
        name=f"{FEATURE_VIEW_NAME}_source",
        path=offline_path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )
    return FeatureView(
        name=FEATURE_VIEW_NAME,
        entities=[household],
        ttl=timedelta(days=ttl_days),
        schema=schema,
        online=False,  # offline history only; serving reads precomputed scores from Redis
        source=source,
        description=f"{len(MODEL_FEATURES)} point-in-time features ({FEATURE_SET_VERSION}), one snapshot per cutoff",
    )
