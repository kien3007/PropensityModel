"""
Feature quality gate: each check catches its kind of bad snapshot, real-looking snapshots pass,
the feature store refuses to write a failing snapshot, and the batch pipeline stops on one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import settings
from src.feature_store.client import FeatureStoreClient
from src.features.contract import NUMERIC_FEATURES
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.features.quality import (
    FeatureQualityError,
    check_feature_quality,
    find_quality_violations,
    numeric_bounds,
)


@pytest.fixture
def builder(synthetic_data):
    return MultiDomainFeatureBuilder(synthetic_data)


@pytest.fixture
def snap(builder):
    return builder.features_at(150)


def test_every_numeric_feature_has_a_bound():
    assert set(numeric_bounds()) == set(NUMERIC_FEATURES)


def test_builder_snapshots_pass(builder):
    previous = None
    for cutoff in (70, 100, 120, 150, 180):
        current = builder.features_at(cutoff)
        assert find_quality_violations(current, previous) == []
        previous = current


@pytest.mark.parametrize(
    "corrupt, expected",
    [
        (lambda df: df.assign(spend_90d=-df["spend_90d"] - 1), "spend_90d: "),
        (lambda df: df.assign(store_loyalty_index=1.5), "store_loyalty_index: "),
        (lambda df: df.assign(days_since_last_basket=500.0), "days_since_last_basket: "),
        (lambda df: df.assign(baskets_count_30d=df["baskets_count_30d"] + 0.5), "baskets_count_30d: "),
        (lambda df: df.assign(classification_2="W"), "classification_2: unseen categories ['W']"),
        (lambda df: df.assign(spend_30d=df["spend_180d"] + 10), "spend_30d > spend_90d"),
        (lambda df: pd.concat([df, df.head(3)], ignore_index=True), "3 duplicated household_key"),
    ],
)
def test_each_check_catches_its_violation(snap, corrupt, expected):
    violations = find_quality_violations(corrupt(snap))
    assert any(v.startswith(expected) for v in violations), violations


def test_coverage_drop_against_previous_snapshot(builder, snap):
    previous = builder.features_at(120)
    shrunk = snap.head(int(len(snap) * 0.5))
    assert any("coverage dropped" in v for v in find_quality_violations(shrunk, previous))


def test_zero_share_jump_against_previous_snapshot(builder, snap):
    previous = builder.features_at(120)
    broken_join = snap.assign(spend_180d=0.0, spend_90d=0.0, spend_30d=0.0)
    violations = find_quality_violations(broken_join, previous)
    assert any(v.startswith("spend_180d: share of zeros") for v in violations)
    # the campaign flag legitimately flips when campaigns start or end
    flipped = snap.assign(is_targeted_by_active_campaign=1 - previous["is_targeted_by_active_campaign"].values)
    assert find_quality_violations(flipped, previous) == []


def test_error_lists_violations_and_cutoff(snap):
    with pytest.raises(FeatureQualityError, match="DAY 150 failed") as exc:
        check_feature_quality(snap.assign(perishable_basket_ratio_90d=2.0), cutoff_day=150)
    assert exc.value.violations


def test_store_refuses_a_failing_snapshot(builder, snap, tmp_path):
    client = FeatureStoreClient(root=tmp_path / "fs")
    client.write_snapshot(builder.features_at(120), 120)
    before = pd.read_parquet(client.offline_path)
    with pytest.raises(FeatureQualityError):
        client.write_snapshot(snap.assign(spend_90d=-5.0), 150)
    assert client.snapshot_days() == [120]
    pd.testing.assert_frame_equal(pd.read_parquet(client.offline_path), before)


def test_gate_can_be_disabled(snap, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FEATURE_QUALITY_CHECKS", False)
    client = FeatureStoreClient(root=tmp_path / "fs")
    client.write_snapshot(snap.assign(classification_2="W"), 150)
    assert client.snapshot_days() == [150]


@pytest.mark.parametrize("store_enabled", [True, False])
def test_batch_stops_on_bad_snapshot_even_without_strict_infra(
    store_enabled, synthetic_raw_dir, synthetic_data, tmp_path, monkeypatch
):
    from src.pipelines.batch_pipeline import run_batch_pipeline
    from src.serving.batch_store import BatchScoreStore

    monkeypatch.setattr(settings, "STRICT_INFRA", False)
    monkeypatch.setattr(settings, "FEATURE_STORE_ENABLED", store_enabled)
    original = MultiDomainFeatureBuilder.features_at
    monkeypatch.setattr(
        MultiDomainFeatureBuilder, "features_at",
        lambda self, cutoff: original(self, cutoff).assign(perishable_basket_ratio_90d=3.0),
    )
    store = BatchScoreStore(redis_client=None)
    with pytest.raises(FeatureQualityError):
        run_batch_pipeline(
            cutoff_day=180, raw_data_dir=synthetic_raw_dir, data=synthetic_data, output_dir=tmp_path / "out",
            lakehouse_path=tmp_path / "lh" / "wh.duckdb", lapse_model_path=tmp_path / "none.joblib",
            batch_store=store,
        )
    assert not list((tmp_path / "out").glob("household_scores_*"))  # nothing was scored or exported
