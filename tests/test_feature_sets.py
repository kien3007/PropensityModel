"""
Versioned feature sets: multidomain_v2 (27 features, exact N-day windows) is the current set,
multidomain_v1 (28 features, windows that also held the day cutoff - N) stays computable so a
v1 champion can still be scored, compared against a v2 challenger and rolled back to.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.contract import FEATURE_SET_VERSION, FEATURES_V1, MODEL_FEATURES, features_for
from src.features.pit_extractor import MultiDomainFeatureBuilder, compute_multidomain_features


def test_feature_sets():
    assert FEATURE_SET_VERSION == "multidomain_v2"
    assert len(features_for("multidomain_v1")) == 28 and "primary_store_id" in FEATURES_V1
    assert len(MODEL_FEATURES) == 27 and "primary_store_id" not in MODEL_FEATURES
    assert [f for f in FEATURES_V1 if f != "primary_store_id"] == MODEL_FEATURES  # same order otherwise


def _baskets(days):
    rows = [{"household_key": 1, "BASKET_ID": i, "DAY": d, "STORE_ID": 7, "basket_spend": 10.0,
             "basket_items": 2, "has_perishable": True} for i, d in enumerate(days)]
    b = pd.DataFrame(rows).sort_values("DAY")
    b["inter_trip_days"] = b["DAY"].diff()
    return b


@pytest.mark.parametrize("version,expected_30d,expected_14d", [
    ("multidomain_v1", 2, 1),   # DAY 70 (= 100 - 30) and DAY 86 (= 100 - 14) still counted
    ("multidomain_v2", 1, 0),   # exactly 30 / 14 days: DAY 71..100 / 87..100
])
def test_window_boundary(version, expected_30d, expected_14d):
    feats = compute_multidomain_features(_baskets([40, 70, 86, 95]), cutoff=100, households=[1],
                                         lookback_days=60, feature_set_version=version)
    row = feats.iloc[0]
    assert row["baskets_count_30d"] == expected_30d + 1  # + the DAY 95 basket
    assert row["baskets_count_14d"] == expected_14d + 1
    assert ("primary_store_id" in feats.columns) == (version == "multidomain_v1")


def test_lookback_boundary():
    v1 = compute_multidomain_features(_baskets([40, 95]), 100, [1], lookback_days=60, feature_set_version="multidomain_v1")
    v2 = compute_multidomain_features(_baskets([40, 95]), 100, [1], lookback_days=60, feature_set_version="multidomain_v2")
    assert v1.iloc[0]["baskets_count_180d"] == 2  # DAY 40 = 100 - 60 is inside v1's lookback
    assert v2.iloc[0]["baskets_count_180d"] == 1


def test_builder_for_version_shares_baskets(synthetic_data):
    v2 = MultiDomainFeatureBuilder(synthetic_data)
    v1 = v2.for_version("multidomain_v1")
    assert v1.baskets is v2.baskets and v2.for_version("multidomain_v2") is v2
    assert list(v1.features_at(150).columns[2:]) == FEATURES_V1
    assert list(v2.features_at(150).columns[2:]) == MODEL_FEATURES


def test_feature_store_keeps_one_history_per_feature_set(synthetic_data, tmp_path):
    from src.feature_store.client import FeatureStoreClient

    builder = MultiDomainFeatureBuilder(synthetic_data)
    v2 = FeatureStoreClient(root=tmp_path / "fs")
    v1 = FeatureStoreClient(root=tmp_path / "fs", feature_set_version="multidomain_v1")
    v2.write_snapshot(builder.features_at(150), 150)
    v1.write_snapshot(builder.for_version("multidomain_v1").features_at(150), 150)
    assert v1.offline_path != v2.offline_path
    assert "primary_store_id" in v1.get_historical_features(builder.households, 150).columns
    assert "primary_store_id" not in v2.get_historical_features(builder.households, 150).columns


def test_v2_challenger_is_compared_against_v1_champion(synthetic_data, small_training_config, tmp_path):
    """The gate scores the v1 champion with its own features on the same test cutoffs."""
    from src.pipelines.train_pipeline import run_training_pipeline

    registry = tmp_path / "registry"
    v1_cfg = small_training_config.with_overrides(feature_set_version="multidomain_v1")
    first = run_training_pipeline(config=v1_cfg, data=synthetic_data, output_dir=tmp_path / "a",
                                  registry_dir=registry, force_promote=True, feature_source="builder")
    assert first["promoted"]
    second = run_training_pipeline(config=small_training_config, data=synthetic_data, output_dir=tmp_path / "b",
                                   registry_dir=registry, promote=False, feature_source="builder")
    comparison = second["governance_gate"]["champion_comparison"]
    assert comparison["status"] == "compared"
    assert comparison["champion_feature_set"] == "multidomain_v1"
    assert comparison["challenger_feature_set"] == "multidomain_v2"
    assert comparison["rows_compared"] > 0


def test_batch_scores_a_v1_champion_with_v1_features(synthetic_raw_dir, synthetic_data, small_training_config, tmp_path):
    from src.pipelines.batch_pipeline import run_batch_pipeline
    from src.pipelines.train_pipeline import run_training_pipeline
    from src.serving.batch_store import BatchScoreStore

    v1_cfg = small_training_config.with_overrides(feature_set_version="multidomain_v1")
    run_training_pipeline(config=v1_cfg, data=synthetic_data, output_dir=tmp_path / "t",
                          registry_dir=tmp_path / "reg", force_promote=True, feature_source="builder")
    summary = run_batch_pipeline(
        cutoff_day=180, raw_data_dir=synthetic_raw_dir, data=synthetic_data, output_dir=tmp_path / "out",
        lakehouse_path=tmp_path / "lh" / "wh.duckdb", model_path=tmp_path / "reg" / "champion_model.joblib",
        lapse_model_path=tmp_path / "none.joblib", batch_store=BatchScoreStore(redis_client=None),
    )
    assert summary["feature_set_version"] == "multidomain_v1" and summary["feature_count"] == 28
    assert summary["total_households_scored"] == len(synthetic_data.households)
