from __future__ import annotations

import numpy as np
import pandas as pd

from propensity_pipeline import (
    DEMOGRAPHIC_COLUMNS,
    TARGET_COLUMN,
    PropensityConfig,
    construct_modeling_table,
    temporal_split,
)


def _demographics(households: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame({"household_key": households})
    for column in DEMOGRAPHIC_COLUMNS:
        frame[column] = "Unknown"
    return frame


def test_snapshot_features_and_label_respect_cutoff() -> None:
    daily = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 3],
            "DAY": [1, 10, 13, 2, 12],
            "spend": [10.0, 20.0, 30.0, 5.0, 8.0],
            "units": [1.0, 2.0, 3.0, 1.0, 1.0],
            "baskets": [1.0, 1.0, 1.0, 1.0, 1.0],
            "discount": [0.0, 2.0, 0.0, 0.0, 0.0],
            "active_days": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    config = PropensityConfig(horizon_days=3, minimum_history_days=1)
    table, _ = construct_modeling_table(
        daily,
        _demographics([1, 2, 3]),
        config,
        snapshots=[10],
    )
    household_1 = table.loc[table["household_key"] == 1].iloc[0]
    assert household_1["spend_7d"] == 20.0
    assert household_1["recency_days"] == 0.0
    assert household_1[TARGET_COLUMN] == 1
    household_2 = table.loc[table["household_key"] == 2].iloc[0]
    assert household_2[TARGET_COLUMN] == 0
    assert 3 not in table["household_key"].to_numpy()


def test_temporal_split_is_ordered_and_has_embargoes() -> None:
    snapshots = np.arange(100, 240, 7)
    table = pd.DataFrame(
        {
            "household_key": np.tile([1, 2], len(snapshots)),
            "snapshot_day": np.repeat(snapshots, 2),
            TARGET_COLUMN: 0,
        }
    )
    train, validation, test, manifest = temporal_split(table, PropensityConfig())
    train_max = int(table.loc[train, "snapshot_day"].max())
    validation_min = int(table.loc[validation, "snapshot_day"].min())
    validation_max = int(table.loc[validation, "snapshot_day"].max())
    test_min = int(table.loc[test, "snapshot_day"].min())
    assert validation_min - train_max == 14
    assert test_min - validation_max == 14
    assert manifest["embargo_snapshot_days"] == [train_max + 7, validation_max + 7]
