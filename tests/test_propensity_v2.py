from __future__ import annotations

import numpy as np
import pandas as pd

from propensity_v2.config import ADDITIVE_DAILY_METRICS, DEMOGRAPHIC_COLUMNS, PropensityV2Config
from propensity_v2.features import build_modeling_table


def _daily() -> pd.DataFrame:
    rows = []
    for household, day, spend, baskets in [
        (1, 1, 10.0, 1.0),
        (1, 10, 20.0, 1.0),
        (1, 13, 30.0, 1.0),
        (2, 2, 5.0, 1.0),
        (3, 12, 8.0, 1.0),
    ]:
        row = {metric: 0.0 for metric in ADDITIVE_DAILY_METRICS}
        row.update(
            {
                "household_key": household,
                "DAY": day,
                "spend": spend,
                "baskets": baskets,
                "line_items": baskets,
                "active_days": 1.0,
                "grocery_spend": spend,
                "morning_baskets": baskets,
                "product_day_diversity": baskets,
                "department_day_diversity": baskets,
                "store_day_diversity": baskets,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _demographics() -> pd.DataFrame:
    frame = pd.DataFrame({"household_key": [1, 2, 3]})
    for column in DEMOGRAPHIC_COLUMNS:
        frame[column] = "Unknown"
    frame["has_demographics"] = 0
    return frame


def test_v2_snapshot_is_point_in_time_and_excludes_future_households() -> None:
    config = PropensityV2Config(horizon_days=3, minimum_history_days=1)
    table, _ = build_modeling_table(
        _daily(), _demographics(), config, snapshots=[10]
    )
    household = table.loc[table["household_key"] == 1].iloc[0]
    assert household["spend_7d"] == 20.0
    assert household[config.target_column] == 1
    assert 3 not in table["household_key"].to_numpy()
    assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()


def test_v2_quantity_is_not_an_additive_feature() -> None:
    assert "units" not in ADDITIVE_DAILY_METRICS
    assert "quantity" not in "|".join(ADDITIVE_DAILY_METRICS).lower()
