"""
Unit tests for Model & Data Drift Monitoring and Prometheus Metrics Exporter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.monitoring.drift_detector import DataDriftDetector
from src.monitoring.metrics import (
    record_drift_metrics,
)
from src.serving.api import app


def test_psi_identical_and_shifted_distributions():
    np.random.seed(42)
    ref = np.random.normal(loc=10.0, scale=2.0, size=1000)

    # 1. Identical distribution: PSI should be very small (< 0.05)
    cur_same = np.random.normal(loc=10.0, scale=2.0, size=1000)
    psi_same = DataDriftDetector.calculate_numerical_psi(ref, cur_same)
    assert psi_same < 0.05

    # 2. Heavily shifted distribution (mean shift from 10 to 18): PSI should be > 0.25
    cur_shifted = np.random.normal(loc=18.0, scale=2.0, size=1000)
    psi_shifted = DataDriftDetector.calculate_numerical_psi(ref, cur_shifted)
    assert psi_shifted >= 0.25


def test_ks_test_p_values():
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=500)

    # Identical: p-value should be high (> 0.05)
    same = np.random.normal(loc=0.0, scale=1.0, size=500)
    ks_same = DataDriftDetector.calculate_ks_test(ref, same)
    assert ks_same["p_value"] > 0.05

    # Drastically shifted: p-value should be very low (< 0.01)
    diff = np.random.normal(loc=3.0, scale=1.0, size=500)
    ks_diff = DataDriftDetector.calculate_ks_test(ref, diff)
    assert ks_diff["p_value"] < 0.01
    assert ks_diff["statistic"] > 0.5


def test_categorical_psi():
    ref = pd.Series(["A", "A", "A", "B", "B", "C"] * 100)
    cur_same = pd.Series(["A", "A", "A", "B", "B", "C"] * 100)
    psi_same = DataDriftDetector.calculate_categorical_psi(ref, cur_same)
    assert psi_same < 0.05

    cur_drifted = pd.Series(["C", "C", "C", "C", "D", "E"] * 100)
    psi_drift = DataDriftDetector.calculate_categorical_psi(ref, cur_drifted)
    assert psi_drift > 0.25


def test_full_drift_audit_report(tmp_path):
    detector = DataDriftDetector(output_dir=tmp_path)

    np.random.seed(42)
    n = 200
    ref_df = pd.DataFrame({
        "days_since_last_basket": np.random.exponential(scale=10, size=n),
        "spend_90d": np.random.normal(loc=500, scale=100, size=n),
        "classification_1": np.random.choice(["19-24", "25-34", "35-44"], size=n),
    })

    # Shifted current dataframe
    cur_df = pd.DataFrame({
        "days_since_last_basket": np.random.exponential(scale=35, size=n),
        "spend_90d": np.random.normal(loc=1200, scale=200, size=n),
        "classification_1": np.random.choice(["55-64", "65+"], size=n),
    })

    ref_scores = np.random.uniform(0.1, 0.4, size=n)
    cur_scores = np.random.uniform(0.7, 0.95, size=n)

    report = detector.generate_full_drift_audit(
        reference_df=ref_df,
        current_df=cur_df,
        reference_scores=ref_scores,
        current_scores=cur_scores,
        run_id="TEST_RUN_01",
    )

    assert report["run_id"] == "TEST_RUN_01"
    assert report["overall_status"] in ["WARNING", "CRITICAL_DRIFT"]
    assert report["requires_retraining"] is True
    assert (tmp_path / "drift_report_TEST_RUN_01.json").exists()
    assert (tmp_path / "drift_report_purchase_latest_full.json").exists()


def test_prometheus_metrics_and_endpoint():
    client = TestClient(app)

    # 1. Trigger prediction to record metrics
    res = client.post("/v1/predict/propensity", json={"household_key": 100})
    assert res.status_code == 200

    # 2. Record drift metric
    record_drift_metrics({"spend_90d": 0.18, "days_since_last_basket": 0.05})

    # 3. Scrape /metrics endpoint
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    metrics_text = metrics_res.text

    assert "propensity_requests_total" in metrics_text
    assert "propensity_prediction_latency_seconds_bucket" in metrics_text
    assert "propensity_served_score_bucket" in metrics_text
    assert "propensity_decile_total" in metrics_text
    assert "propensity_urgency_tier_total" in metrics_text
    assert "propensity_feature_drift_psi" in metrics_text
