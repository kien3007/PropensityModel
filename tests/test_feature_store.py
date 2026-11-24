"""
Feast historical feature store: append-only snapshot history, point-in-time reads that equal
the feature builder, TTL behaviour, online store freshness, training through the store and
the batch pipeline write path.
"""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from src.config.settings import settings
from src.feature_store.client import FeatureStoreClient, FeatureStoreSource, MissingSnapshotError
from src.feature_store.definitions import FEATURE_VIEW_NAME, day_to_timestamp, timestamp_to_day
from src.features.contract import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.training.dataset_builder import DatasetBuilder


@pytest.fixture
def builder(synthetic_data):
    return MultiDomainFeatureBuilder(synthetic_data)


@pytest.fixture
def client(tmp_path):
    return FeatureStoreClient(root=tmp_path / "fs")


def test_schema_is_generated_from_contract(client):
    fields = {f.name: str(f.dtype) for f in client.feature_view.schema}
    assert set(fields) == set(MODEL_FEATURES)  # Feast orders fields itself; reads are by name
    assert {fields[c] for c in CATEGORICAL_FEATURES} == {"String"}
    assert {fields[c] for c in NUMERIC_FEATURES} == {"Float64"}
    assert client.feature_view.name == FEATURE_VIEW_NAME


def test_timeline_roundtrip():
    assert timestamp_to_day(day_to_timestamp(711)) == 711


def test_snapshots_accumulate_and_replace_by_cutoff(client, builder):
    client.backfill(builder, [120, 150])
    client.write_snapshot(builder.features_at(180), 180)
    assert client.snapshot_days() == [120, 150, 180]
    # re-writing a cutoff replaces it instead of duplicating rows
    client.write_snapshot(builder.features_at(150), 150, online=False)
    offline = pd.read_parquet(client.offline_path)
    assert client.snapshot_days() == [120, 150, 180]
    assert len(offline) == 3 * len(builder.households)
    assert not offline.duplicated(["household_key", "cutoff_day"]).any()
    assert client.backfill(builder, [120, 150, 180]) == []  # nothing missing


def test_point_in_time_read_equals_builder(client, builder):
    client.backfill(builder, [120, 150, 180])
    for cutoff in (120, 150, 180):
        got = client.get_historical_features(builder.households, cutoff)
        expected = builder.features_at(cutoff)
        pdt.assert_frame_equal(got[["household_key", *MODEL_FEATURES]], expected[["household_key", *MODEL_FEATURES]])


def test_pit_join_uses_latest_snapshot_within_ttl(client, builder):
    client.backfill(builder, [150])
    at_153 = client.get_historical_features(builder.households, 153)  # 3 days after the snapshot
    pdt.assert_frame_equal(at_153[MODEL_FEATURES], builder.features_at(150)[MODEL_FEATURES].reset_index(drop=True))
    with pytest.raises(MissingSnapshotError, match="no snapshot within 7 days"):
        client.get_historical_features(builder.households, 165)  # snapshot is stale
    with pytest.raises(MissingSnapshotError):
        client.get_historical_features(builder.households, 140)  # no snapshot at or before


def test_online_store_holds_latest_snapshot(client, builder):
    client.write_snapshot(builder.features_at(180), 180)
    client.write_snapshot(builder.features_at(100), 100)  # older: history only, online untouched
    online = client.get_online_features([1, 2]).set_index("household_key")
    latest = builder.features_at(180).set_index("household_key")
    assert online.loc[[1, 2], "spend_90d"].tolist() == latest.loc[[1, 2], "spend_90d"].tolist()


def test_training_through_feature_store_matches_builder(builder, small_training_config, tmp_path):
    source = FeatureStoreSource(builder, FeatureStoreClient(root=tmp_path / "fs"))
    via_store = DatasetBuilder(source).build_splits(small_training_config)
    direct = DatasetBuilder(builder).build_splits(small_training_config)
    for name in ("fit", "val", "test"):
        pdt.assert_frame_equal(via_store[name].frame, direct[name].frame)
    cutoffs = {*small_training_config.fit_cutoffs, *small_training_config.val_cutoffs, *small_training_config.test_cutoffs}
    assert set(source.client.snapshot_days()) == cutoffs  # backfilled once


def test_source_without_backfill_refuses_missing_cutoffs(builder, tmp_path):
    source = FeatureStoreSource(builder, FeatureStoreClient(root=tmp_path / "fs"), backfill_missing=False)
    with pytest.raises(MissingSnapshotError):
        source.features_at(150)


def test_train_pipeline_reads_from_feature_store(synthetic_data, small_training_config, tmp_path):
    from src.pipelines.train_pipeline import run_training_pipeline

    feast_run = run_training_pipeline(config=small_training_config, data=synthetic_data, output_dir=tmp_path / "a",
                                      registry_dir=tmp_path / "ra", promote=False, feature_source="feast")
    direct_run = run_training_pipeline(config=small_training_config, data=synthetic_data, output_dir=tmp_path / "b",
                                       registry_dir=tmp_path / "rb", promote=False, feature_source="builder")
    assert feast_run["feature_source"] == "feast" and direct_run["feature_source"] == "builder"
    assert feast_run["test_metrics"]["roc_auc"] == pytest.approx(direct_run["test_metrics"]["roc_auc"], abs=1e-12)
    assert FeatureStoreClient().snapshot_days()  # default store (isolated per test) was populated


def test_batch_pipeline_appends_snapshot_and_refreshes_online(synthetic_raw_dir, synthetic_data, tmp_path):
    from src.pipelines.batch_pipeline import run_batch_pipeline
    from src.serving.batch_store import BatchScoreStore

    summary = run_batch_pipeline(
        cutoff_day=180, raw_data_dir=synthetic_raw_dir, data=synthetic_data, output_dir=tmp_path / "out",
        lakehouse_path=tmp_path / "lh" / "wh.duckdb", lapse_model_path=tmp_path / "none.joblib",
        batch_store=BatchScoreStore(redis_client=None),
    )
    fs = FeatureStoreClient(root=settings.FEATURE_STORE_DIR)
    assert summary["feature_store"]["rows"] == len(synthetic_data.households)
    assert fs.snapshot_days() == [180]
    assert fs.get_online_features([1])["spend_90d"].notna().all()
