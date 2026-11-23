"""
Unit tests for Ground-Truth Feedback Loop and Prospective Attribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.lakehouse.engine import LakehouseEngine
from src.monitoring.feedback_loop import GroundTruthFeedbackLoop


@pytest.fixture
def temp_lakehouse(tmp_path):
    db_path = tmp_path / "lakehouse" / "test_warehouse.duckdb"
    engine = LakehouseEngine(lakehouse_path=db_path)
    return engine


def test_ground_truth_attribution_mock_evaluation(temp_lakehouse, tmp_path):
    loop = GroundTruthFeedbackLoop(lakehouse_engine=temp_lakehouse, output_dir=tmp_path)

    # Synthetic historical predictions for 100 households
    np.random.seed(42)
    n = 100
    probs = np.random.uniform(0.1, 0.9, size=n)
    deciles = np.digitize(probs, np.linspace(0, 1, 11))
    deciles = np.clip(deciles, 1, 10)

    preds_df = pd.DataFrame({
        "household_key": list(range(1, n + 1)),
        "purchase_probability": probs,
        "decile": deciles,
        "urgency_tier": ["REGULAR"] * n,
    })

    # Synthetic transactions during horizon [101, 114]
    tx_file = tmp_path / "synthetic_tx.csv"
    tx_df = pd.DataFrame({
        "household_key": [1, 2, 3, 10, 20, 30, 50, 99],
        "BASKET_ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        "DAY": [102, 105, 110, 112, 103, 108, 114, 106],
        "SALES_VALUE": [25.0, 45.0, 12.0, 60.0, 80.0, 30.0, 15.0, 90.0],
    })
    tx_df.to_csv(tx_file, index=False)

    result = loop.evaluate_realized_performance(
        cutoff_day=100,
        horizon_days=14,
        predictions_df=preds_df,
        raw_tx_csv=tx_file,
    )

    assert result["cutoff_day"] == 100
    assert result["horizon_days"] == 14
    assert result["sample_size"] == n
    assert 0.0 <= result["realized_roc_auc"] <= 1.0
    assert 0.0 <= result["realized_brier_score"] <= 1.0
    assert result["attribution_status"] in ["EXCELLENT", "HEALTHY", "DEGRADED", "CRITICAL_DRIFT"]
    assert len(result["decile_breakdown"]) == 10

    # Verify DuckDB persistence
    con = temp_lakehouse.get_connection()
    try:
        saved_records = con.execute("SELECT COUNT(*) FROM fct_model_attribution;").fetchone()[0]
        assert saved_records == 1
    finally:
        con.close()

    # Verify JSON report was created
    assert (tmp_path / "feedback_attribution_day_100.json").exists()
