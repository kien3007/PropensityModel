"""
Feature contract tests: point-in-time correctness of the 28 multi-domain features and
train/serve parity (CSV vs Lakehouse loaders, training vs batch-scoring code paths).
"""
import pandas as pd
import pandas.testing as pdt
import pytest

from src.data.loader import DataLoader, RawDataBundle
from src.features.contract import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    FeatureContractError,
    infer_feature_set_version,
    validate_feature_frame,
)
from src.features.pit_extractor import MultiDomainFeatureBuilder, PointInTimeFeatureExtractor
from src.lakehouse.engine import LakehouseEngine
from src.training.dataset_builder import DatasetBuilder


def test_features_ignore_rows_after_cutoff(synthetic_tables):
    tx = synthetic_tables["transactions"]
    cutoff = 120
    full = RawDataBundle(transactions=tx, products=synthetic_tables["products"])
    truncated = RawDataBundle(transactions=tx[tx["DAY"] <= cutoff], products=synthetic_tables["products"])
    a = MultiDomainFeatureBuilder(full).features_at(cutoff)
    b = MultiDomainFeatureBuilder(truncated).features_at(cutoff)
    # Same universe so rows align; future rows must not change any feature value.
    b = b.set_index("household_key").reindex(a["household_key"]).reset_index()
    numeric = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]
    pdt.assert_frame_equal(a[numeric].fillna(-1), b[numeric].fillna(-1), check_dtype=False)


def test_feature_values_on_known_household(synthetic_data):
    builder = MultiDomainFeatureBuilder(synthetic_data)
    cutoff = 100
    feats = builder.features_at(cutoff).set_index("household_key")
    b = builder.baskets
    hh = int(b["household_key"].iloc[0])
    hist = b[(b["household_key"] == hh) & (b["DAY"] <= cutoff) & (b["DAY"] >= cutoff - 180)]
    row = feats.loc[hh]
    assert row["days_since_last_basket"] == cutoff - hist["DAY"].max()
    assert row["baskets_count_30d"] == hist[hist["DAY"] >= cutoff - 30]["BASKET_ID"].nunique()
    assert row["spend_90d"] == pytest.approx(hist[hist["DAY"] >= cutoff - 90]["basket_spend"].sum())


def test_feature_frame_satisfies_contract(synthetic_data):
    feats = MultiDomainFeatureBuilder(synthetic_data).features_at(150)
    validate_feature_frame(feats)
    assert len(feats) == len(synthetic_data.households)
    for col in CATEGORICAL_FEATURES:
        assert feats[col].dtype.name == "category"
    # Households without demographics fall back to UNKNOWN, not NaN.
    assert "UNKNOWN" in set(feats["classification_1"].astype(str))


def test_contract_violations_are_reported(synthetic_data):
    feats = MultiDomainFeatureBuilder(synthetic_data).features_at(150)
    with pytest.raises(FeatureContractError, match="missing"):
        validate_feature_frame(feats.drop(columns=["spend_90d"]))
    with pytest.raises(FeatureContractError):
        infer_feature_set_version(MODEL_FEATURES[:-1])


def test_csv_and_lakehouse_loaders_produce_identical_features(synthetic_raw_dir, tmp_path):
    """Train/serve skew guard: both data sources must yield byte-identical features."""
    from_csv = DataLoader.from_csv(synthetic_raw_dir)
    lakehouse = LakehouseEngine(lakehouse_path=tmp_path / "lh" / "wh.duckdb", raw_data_dir=synthetic_raw_dir)
    from_lake = DataLoader.from_lakehouse(lakehouse)

    assert from_csv.fingerprint() == from_lake.fingerprint()
    for cutoff in (90, 150):
        a = MultiDomainFeatureBuilder(from_csv).features_at(cutoff)
        b = MultiDomainFeatureBuilder(from_lake).features_at(cutoff)
        pdt.assert_frame_equal(a, b)


def test_training_rows_equal_batch_scoring_rows(synthetic_data):
    """The features a model is trained on for cutoff t equal what batch scoring computes at t."""
    builder = MultiDomainFeatureBuilder(synthetic_data)
    cutoff = 150
    train_rows = DatasetBuilder(builder).build([cutoff], horizon_days=7)
    batch_rows = PointInTimeFeatureExtractor.extract_multidomain_features(cutoff, data=synthetic_data)
    pdt.assert_frame_equal(
        train_rows[MODEL_FEATURES].reset_index(drop=True),
        batch_rows[MODEL_FEATURES].reset_index(drop=True),
        check_categorical=False,
    )
    pd.testing.assert_series_equal(train_rows["household_key"], batch_rows["household_key"])
