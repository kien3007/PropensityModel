"""
Unit tests for Feast Feature Store integration.
"""
from __future__ import annotations

import datetime
from pathlib import Path
import pandas as pd
import pytest

from src.feature_store.client import FeastFeatureStoreClient
from src.features.pit_extractor import MODEL_FEATURES


@pytest.fixture(scope="module")
def feast_client():
    client = FeastFeatureStoreClient()
    return client


def test_feast_feature_views_registered(feast_client):
    views = feast_client.get_feature_view_names()
    assert "household_features_view" in views


def test_feast_materialize_and_online_retrieval(feast_client, tmp_path):
    # Materialize snapshot to Feast Online Store using cutoff 711
    # Check if raw data exists
    from src.config.settings import settings
    raw_dir = settings.DATA_RAW_DIR
    if not (raw_dir / "transaction_data.csv").exists():
        pytest.skip("Raw Dunnhumby data not found, skipping live materialization.")

    parquet_file = feast_client.materialize_snapshot(cutoff_day=711)
    assert parquet_file.exists()

    # Query online features for 3 households
    online_features = feast_client.get_online_features(household_keys=[1, 2, 100])
    assert "household_key" in online_features
    assert len(online_features["household_key"]) == 3
    # Check that key features exist in online response
    assert "days_since_last_basket" in online_features
    assert "spend_90d" in online_features


def test_feast_historical_point_in_time_join(feast_client):
    from src.config.settings import settings
    raw_dir = settings.DATA_RAW_DIR
    gold_parquet = settings.BASE_DIR / "data" / "lakehouse" / "gold" / "fct_household_features_snapshot.parquet"
    if not gold_parquet.exists():
        pytest.skip("Gold snapshot parquet not found, skipping historical PIT join.")

    # Entity dataframe with event timestamp matching cutoff day 711
    base_date = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    event_time = base_date + datetime.timedelta(days=711)

    entity_df = pd.DataFrame({
        "household_key": [1, 2, 100],
        "event_timestamp": [event_time, event_time, event_time],
    })

    features = [
        "household_features_view:days_since_last_basket",
        "household_features_view:spend_90d",
        "household_features_view:urgency_ratio",
    ]

    hist_df = feast_client.get_historical_features(entity_df=entity_df, feature_refs=features)
    assert len(hist_df) == 3
    assert "days_since_last_basket" in hist_df.columns
    assert "spend_90d" in hist_df.columns
    assert "urgency_ratio" in hist_df.columns
