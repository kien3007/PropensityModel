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

from src.features.contract import CATEGORICAL_FEATURES, FEATURE_SET_VERSION, features_for

FEATURE_EPOCH = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def feature_view_name(feature_set_version: str = FEATURE_SET_VERSION) -> str:
    return f"household_features_{feature_set_version}"


FEATURE_VIEW_NAME = feature_view_name()  # view of the current feature set
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


def build_feature_view(offline_path: str, ttl_days: int, feature_set_version: str = FEATURE_SET_VERSION) -> FeatureView:
    features = features_for(feature_set_version)
    view = feature_view_name(feature_set_version)
    schema = [
        Field(name=f, dtype=String if f in CATEGORICAL_FEATURES else Float64)
        for f in features
    ]
    source = FileSource(
        name=f"{view}_source",
        path=offline_path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )
    return FeatureView(
        name=view,
        entities=[household],
        ttl=timedelta(days=ttl_days),
        schema=schema,
        online=False,  # offline history only; serving reads precomputed scores from Redis
        source=source,
        description=f"{len(features)} point-in-time features ({feature_set_version}), one snapshot per cutoff",
    )
