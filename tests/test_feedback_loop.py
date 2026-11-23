"""
Ground-truth feedback, retraining policy, drift-triggered retraining, fusion-rule evaluation
and cross-process monitoring gauges.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.loader import RawDataBundle
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.lakehouse.engine import LakehouseEngine
from src.models.predictor import PropensityPredictor
from src.monitoring.drift_detector import DataDriftDetector
from src.monitoring.feedback_loop import GroundTruthFeedbackLoop
from src.monitoring.reference import build_champion_reference
from src.monitoring.retraining_policy import decide_retraining
from src.monitoring.state import MonitoringState
from src.pipelines.train_pipeline import run_training_pipeline


@pytest.fixture(scope="module")
def synthetic_champion(tmp_path_factory, synthetic_data):
    from src.config.model_config import TrainingConfig

    cfg = TrainingConfig(
        horizon_days=7, lookback_days=60,
        fit_cutoffs=(70, 80, 90, 100, 110, 120), val_cutoffs=(135, 145), test_cutoffs=(165, 175),
        lgbm_params={"objective": "binary", "n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7,
                     "min_child_samples": 5, "random_state": 42, "n_jobs": 1, "verbosity": -1},
        early_stopping_rounds=10, calibration_cv_folds=3,
    )
    out = tmp_path_factory.mktemp("champion")
    run_training_pipeline(config=cfg, data=synthetic_data, output_dir=out / "run", registry_dir=out / "reg",
                          force_promote=True)
    return out / "reg" / "champion_model.joblib"


@pytest.fixture
def loop(tmp_path, synthetic_data):
    lakehouse = LakehouseEngine(lakehouse_path=tmp_path / "lh" / "wh.duckdb")
    state = MonitoringState(redis_client=None, report_dir=tmp_path / "reports")
    return GroundTruthFeedbackLoop(lakehouse_engine=lakehouse, output_dir=tmp_path / "reports",
                                   data=synthetic_data, state=state)


def test_feedback_replays_latest_matured_cutoff(loop, synthetic_champion, synthetic_data):
    result = loop.evaluate_realized_performance(target="purchase", model_path=synthetic_champion)
    last_day = int(synthetic_data.transactions["DAY"].max())
    assert result["cutoff_day"] == last_day - 7 and result["horizon_days"] == 7
    assert result["prediction_source"] == "replay"
    assert result["sample_size"] == len(synthetic_data.households)
    assert np.isfinite(result["expected_roc_auc"]) and "auc_drop" in result
    assert loop.state.latest("feedback_purchase")["eval_id"] == result["eval_id"]

    con = loop.lakehouse.get_connection()
    try:
        row = con.execute("SELECT target, realized_roc_auc, expected_roc_auc FROM fct_model_attribution").fetchall()
    finally:
        con.close()
    assert row[0][0] == "purchase" and row[0][1] == pytest.approx(result["realized_roc_auc"])


def test_feedback_uses_stored_batch_predictions(loop, synthetic_champion, synthetic_data):
    cutoff = 150
    feats = MultiDomainFeatureBuilder(synthetic_data).features_at(cutoff)
    scores = PropensityPredictor(model_path=synthetic_champion).score_households(feats)
    loop.lakehouse.build_gold_layer()  # creates fct_weekly_batch_scores
    loop.lakehouse.save_batch_scores(scores, run_id="B150", snapshot_day=cutoff)
    result = loop.evaluate_realized_performance(target="purchase", cutoff_day=cutoff, model_path=synthetic_champion)
    assert result["prediction_source"] == "batch"


def test_degradation_is_detected_and_triggers_retraining(loop, synthetic_champion, synthetic_data):
    rng = np.random.default_rng(3)
    hh = synthetic_data.households
    noise = pd.DataFrame({"household_key": hh, "purchase_probability": rng.uniform(0, 1, len(hh)),
                          "decile": rng.integers(1, 11, len(hh))})
    result = loop.evaluate_realized_performance(target="purchase", cutoff_day=150, predictions_df=noise,
                                                model_path=synthetic_champion)
    assert result["degraded"] is True and result["attribution_status"] == "DEGRADED"
    decision = decide_retraining(feedback_reports=[result])
    assert decision["retrain"] is True and decision["targets"] == ["purchase"]


def test_immature_labels_are_rejected(loop, synthetic_champion, synthetic_data):
    last_day = int(synthetic_data.transactions["DAY"].max())
    with pytest.raises(ValueError, match="mature"):
        loop.evaluate_realized_performance(target="purchase", cutoff_day=last_day - 2, model_path=synthetic_champion)


def test_old_attribution_table_is_upgraded(tmp_path, synthetic_data):
    lakehouse = LakehouseEngine(lakehouse_path=tmp_path / "old" / "wh.duckdb")
    con = lakehouse.get_connection()
    con.execute("CREATE TABLE fct_model_attribution (eval_id VARCHAR PRIMARY KEY, cutoff_day INTEGER)")
    con.close()
    GroundTruthFeedbackLoop(lakehouse_engine=lakehouse, output_dir=tmp_path, data=synthetic_data,
                            state=MonitoringState(redis_client=None, report_dir=tmp_path))
    con = lakehouse.get_connection()
    cols = {r[0] for r in con.execute("DESCRIBE fct_model_attribution").fetchall()}
    con.close()
    assert {"target", "expected_roc_auc", "auc_drop"} <= cols


def test_simulated_drift_is_critical_and_retrains(synthetic_champion, synthetic_tables, synthetic_data):
    """Acceptance: shifted data -> CRITICAL drift -> scheduler launches retraining."""
    predictor = PropensityPredictor(model_path=synthetic_champion)
    reference = build_champion_reference(predictor, MultiDomainFeatureBuilder(synthetic_data))

    tx = synthetic_tables["transactions"].copy()
    late = tx["DAY"] > 120
    tx.loc[late, "SALES_VALUE"] *= 6.0                      # spend shock
    keep = ~late | (tx["BASKET_ID"] % 3 == 0)               # two thirds of late baskets disappear
    shifted = RawDataBundle(transactions=tx[keep], products=synthetic_tables["products"])
    current = MultiDomainFeatureBuilder(shifted).features_at(180)

    report = DataDriftDetector(output_dir=MonitoringState(redis_client=None).report_dir).generate_full_drift_audit(
        reference_df=reference.features, current_df=current,
        reference_scores=reference.scores, current_scores=predictor.predict_proba(current), run_id="SIM_DRIFT",
    )
    assert report["overall_status"] == "CRITICAL_DRIFT" and report["requires_retraining"] is True

    from src.scheduler.daemon import MLSystemScheduler

    scheduler = MLSystemScheduler()
    retrain_result = {"promoted": False, "lapse": {"promoted": False}}
    with patch("src.scheduler.daemon.run_batch_pipeline", return_value={
            "total_households_scored": 1, "mean_propensity": 0.5, "duration_seconds": 0}), \
         patch.object(MonitoringState, "latest", return_value=report), \
         patch.object(scheduler, "job_monthly_retraining", return_value=retrain_result) as retrain:
        summary = scheduler.job_weekly_batch()
    retrain.assert_called_once()
    assert summary["retraining_decision"]["retrain"] is True
    assert "drift" in retrain.call_args.kwargs["reasons"][0]


def test_no_retraining_when_healthy():
    assert decide_retraining(drift_report={"requires_retraining": False},
                             feedback_reports=[{"degraded": False}])["retrain"] is False


def test_fusion_rule_evaluation(loop, synthetic_data, synthetic_champion):
    cutoff = 150
    labels = MultiDomainFeatureBuilder(synthetic_data).labels_at(cutoff, 7)
    buyers = labels.loc[labels["target"] == 1, "household_key"].tolist()[:10]
    rows = [{"served_at": i, "household_key": hh, "batch_snapshot_day": cutoff, "batch_probability": 0.8,
             "unified_probability": 0.2, "rules": ["intraday_purchase_dampening"]} for i, hh in enumerate(buyers)]
    rows += [{"served_at": 100, "household_key": buyers[0], "batch_snapshot_day": cutoff + 1000,
              "batch_probability": 0.5, "unified_probability": 0.5, "rules": []}]  # not matured: dropped
    result = loop.evaluate_fusion_rules(pd.DataFrame(rows), horizon_days=7)
    rule = result["rules"]["intraday_purchase_dampening"]
    assert rule["n"] == 10 and rule["realized_rate"] == 1.0
    assert rule["rule_improves_calibration"] is False       # these households bought anyway
    assert rule["implied_dampening_factor"] == pytest.approx(1.25)
    assert loop.state.latest("fusion_rules")["rows"] == 10


def test_metrics_endpoint_exposes_cross_process_monitoring(tmp_path, monkeypatch):
    from src.serving import api

    state = MonitoringState(redis_client=None, report_dir=tmp_path)
    state.publish("feedback_purchase", {"realized_roc_auc": 0.80, "expected_roc_auc": 0.86, "auc_drop": 0.06,
                                        "cutoff_day": 697})
    state.publish("drift_purchase", {"overall_status": "CRITICAL_DRIFT", "feature_drift_summary": {"drift_ratio": 0.3},
                                     "prediction_drift": {"psi": 0.4}, "feature_details": {"spend_90d": {"psi": 0.5}}})
    monkeypatch.setattr(api, "monitoring_state", state)
    text = TestClient(api.app).get("/metrics").text
    assert 'propensity_realized_auc_drop{target="purchase"} 0.06' in text
    assert 'propensity_drift_status{target="purchase"} 2.0' in text
    assert 'propensity_feature_drift_psi{feature_name="spend_90d"} 0.5' in text
    assert "propensity_batch_view_households" in text
