"""
Integration and Unit tests for FastAPI Serving Layer API.
Tests /health, /v1/predict/propensity, /v1/speed/event, /v1/batch/scores, and /v1/metrics.
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from src.serving.api import app, fusion_engine


@pytest.fixture(autouse=True)
def reset_engine_state():
    """Ensures clean state between tests."""
    fusion_engine.speed_processor.reset_intraday_deltas()
    fusion_engine._batch_cache.clear()
    yield
    fusion_engine.speed_processor.reset_intraday_deltas()
    fusion_engine._batch_cache.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["service"] == "propensity-serving-api"
    assert "model_bundle" in data
    assert data["feature_count"] in (23, 28)


def test_predict_propensity_default_baseline(client):
    response = client.post("/v1/predict/propensity", json={"household_key": 100})
    assert response.status_code == 200
    data = response.json()
    assert data["household_key"] == 100
    assert 0.0 <= data["unified_propensity"] <= 1.0
    assert data["batch_baseline_propensity"] == 0.50
    assert data["decile"] == 5
    assert data["urgency_tier"] == "REGULAR"
    assert data["applied_fusion_rules"] == []


def test_predict_propensity_validation_error(client):
    # household_key out of range [1, 2500]
    response = client.post("/v1/predict/propensity", json={"household_key": 9999})
    assert response.status_code == 422

    response_negative = client.post("/v1/predict/propensity", json={"household_key": 0})
    assert response_negative.status_code == 422


def test_speed_event_ingestion_and_fusion(client):
    # 1. Ingest POS checkout event
    event_payload = {
        "household_key": 42,
        "event_type": "pos_checkout",
        "sales_value": 85.50,
    }
    event_resp = client.post("/v1/speed/event", json=event_payload)
    assert event_resp.status_code == 202
    assert event_resp.json()["status"] == "ACCEPTED"
    assert event_resp.json()["updated_delta"]["intraday_trips_detected"] == 1

    # 2. Check prediction reflects dampening rule
    pred_resp = client.post("/v1/predict/propensity", json={"household_key": 42})
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["unified_propensity"] == pytest.approx(0.50 * 0.25)
    assert pred_data["urgency_tier"] == "LOW"
    assert pred_data["marketing_action"] == "POST_PURCHASE_COOLDOWN"
    assert "intraday_purchase_dampening" in pred_data["applied_fusion_rules"]


def test_speed_event_coupon_surge(client):
    # Ingest coupon clip
    event_payload = {
        "household_key": 77,
        "event_type": "coupon_clip",
        "coupon_id": 12345,
    }
    client.post("/v1/speed/event", json=event_payload)

    pred_resp = client.post("/v1/predict/propensity", json={"household_key": 77})
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["unified_propensity"] == pytest.approx(0.50 + 0.15)
    assert pred_data["urgency_tier"] == "URGENT"
    assert "coupon_clipping_intent_surge" in pred_data["applied_fusion_rules"]


def test_batch_scores_endpoint(client):
    # Populate mock cache
    mock_batch = pd.DataFrame({
        "household_key": [1, 2, 3],
        "purchase_probability": [0.95, 0.65, 0.15],
        "rank": [1, 800, 2400],
        "decile": [10, 7, 1],
        "urgency_tier": ["URGENT", "REGULAR", "LOW"],
        "marketing_action": ["VIP_COUPON_HALO_UPSELL", "CYCLE_ACCELERATION_NUDGE", "WINBACK_OR_HOLDOUT"],
    })
    fusion_engine.set_batch_cache(mock_batch)

    # Fetch all cached scores
    resp = client.get("/v1/batch/scores")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_available"] == 3
    assert data["returned"] == 3

    # Filter by decile 10
    resp_filtered = client.get("/v1/batch/scores?decile=10")
    assert resp_filtered.status_code == 200
    filtered_data = resp_filtered.json()
    assert filtered_data["returned"] == 1
    assert filtered_data["scores"][0]["household_key"] == 1


def test_metrics_endpoint(client):
    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "cached_households" in data
    assert "active_speed_deltas" in data
    assert len(data["model_features"]) in (23, 28)


def test_model_champion_and_reload_endpoints(client):
    resp = client.get("/v1/models/champion")
    assert resp.status_code == 200
    data = resp.json()
    assert "bundle_name" in data
    assert data["feature_count"] in (23, 28)
    assert len(data["features"]) in (23, 28)

    # Hot-reload endpoint
    reload_resp = client.post("/v1/models/reload")
    assert reload_resp.status_code == 200
    reload_data = reload_resp.json()
    assert reload_data["status"] == "RELOADED"
    assert reload_data["feature_count"] in (23, 28)
