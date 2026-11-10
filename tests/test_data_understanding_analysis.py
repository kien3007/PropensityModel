import pandas as pd

import analyze_propensity_data_understanding as analysis


def test_target_scenarios_use_only_eligible_households_and_future_window(monkeypatch):
    monkeypatch.setattr(analysis, "SNAPSHOTS", (10,))
    basket = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 2, 3],
            "BASKET_ID": [101, 102, 103, 201, 202, 301],
            "DAY": [10, 11, 18, 2, 17, 11],
            "sales": [2.0, 0.0, 5.0, 2.0, 3.0, 4.0],
            "positive_basket": [True, False, True, True, True, True],
            "grocery_department": [True, False, True, False, True, True],
            "fuel_basket": [False, True, False, False, False, False],
            "coupon_basket": [False, False, False, False, True, False],
        }
    )
    panel = analysis.make_panel(basket, max_day=20).set_index("household_key")
    assert set(panel.index) == {1, 2}  # Household 3 first appears after cutoff.
    assert panel.loc[1, "target_any_basket"] == 1
    assert panel.loc[1, "target_positive_basket"] == 0
    assert panel.loc[1, "target_grocery_department"] == 0
    assert panel.loc[2, "target_any_basket"] == 1  # Day 17 is inclusive.
    assert panel.loc[2, "target_positive_basket"] == 1
    assert panel.loc[2, "target_grocery_department"] == 1
    assert panel.loc[2, "coupon_baskets_90d"] == 0  # Day 17 is future.
