"""
Feast FeatureView definitions for Household Purchase Propensity Platform.
Includes 28 multi-domain Point-in-Time features.
"""
from datetime import timedelta
from pathlib import Path
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

try:
    from entities import household
except (ImportError, ModuleNotFoundError):
    from feature_repo.entities import household

PARQUET_PATH = str(Path(__file__).resolve().parent.parent / "data" / "lakehouse" / "gold" / "fct_household_features_snapshot.parquet").replace("\\", "/")

household_features_source = FileSource(
    name="household_features_source",
    path=PARQUET_PATH,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

household_features_view = FeatureView(
    name="household_features_view",
    entities=[household],
    ttl=timedelta(days=365),
    schema=[
        Field(name="days_since_last_basket", dtype=Float32),
        Field(name="days_since_last_perishable", dtype=Float32),
        Field(name="median_inter_trip", dtype=Float32),
        Field(name="std_inter_trip", dtype=Float32),
        Field(name="urgency_ratio", dtype=Float32),
        Field(name="baskets_count_14d", dtype=Float32),
        Field(name="baskets_count_30d", dtype=Float32),
        Field(name="baskets_count_60d", dtype=Float32),
        Field(name="baskets_count_90d", dtype=Float32),
        Field(name="baskets_count_180d", dtype=Float32),
        Field(name="frequency_momentum_30_vs_90", dtype=Float32),
        Field(name="spend_30d", dtype=Float32),
        Field(name="spend_90d", dtype=Float32),
        Field(name="spend_180d", dtype=Float32),
        Field(name="aov_90d", dtype=Float32),
        Field(name="avg_items_per_basket_90d", dtype=Float32),
        Field(name="perishable_basket_ratio_90d", dtype=Float32),
        Field(name="is_targeted_by_active_campaign", dtype=Float32),
        Field(name="coupon_redemption_count_180d", dtype=Float32),
        Field(name="primary_store_id", dtype=Float32),
        Field(name="store_loyalty_index", dtype=Float32),
        Field(name="classification_1", dtype=String),
        Field(name="classification_2", dtype=String),
        Field(name="classification_3", dtype=String),
        Field(name="HOMEOWNER_DESC", dtype=String),
        Field(name="classification_4", dtype=String),
        Field(name="classification_5", dtype=String),
        Field(name="KID_CATEGORY_DESC", dtype=String),
    ],
    online=True,
    source=household_features_source,
    description="28 Point-in-Time behavioral, cadence, loyalty and demographic features for propensity scoring",
)
