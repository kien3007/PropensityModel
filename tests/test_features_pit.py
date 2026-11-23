"""
Unit tests for Point-in-Time Feature Extraction.
Verifies strictly DAY <= cutoff, no data leakage, and feature completeness.
"""
import pandas as pd

from src.features.contract import LEGACY_FEATURES_23
from src.features.pit_extractor import PointInTimeFeatureExtractor


def test_pit_extractor_excludes_future():
    # Synthetic transactions
    tx = pd.DataFrame({
        "household_key": [1, 1, 1, 2],
        "BASKET_ID": [101, 102, 103, 201],
        "DAY": [1, 10, 11, 11],
        "SALES_VALUE": [10.0, 5.0, 20.0, 15.0],
    })

    cutoff = 10
    snapshot = PointInTimeFeatureExtractor.extract_features(tx, cutoff=cutoff, feature_cols=LEGACY_FEATURES_23)

    # Check 2,500 households returned
    assert len(snapshot) == 2500
    assert not snapshot.duplicated(["household_key", "snapshot_day"]).any()

    # Household 1 had trips on day 1 and 10 -> recency = 10 - 10 = 0
    hh1 = snapshot.loc[snapshot["household_key"] == 1].iloc[0]
    assert hh1["recency_days"] == 0.0
    assert hh1["baskets_7d"] == 1  # only basket on day 10 falls in (10-7, 10]
    assert hh1["sales_7d"] == 5.0  # day 11 must be excluded!

    # Household 2 had trip on day 11 -> must be treated as inactive on day 10
    hh2 = snapshot.loc[snapshot["household_key"] == 2].iloc[0]
    assert hh2["baskets_7d"] == 0
    assert hh2["sales_7d"] == 0.0
