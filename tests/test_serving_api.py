"""
Serving API tests: batch-view serving, cold start, fusion rules, API-key auth, rate limiting,
event idempotency, readiness and canary variant serving.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config.settings import settings
from src.serving.api import app, fusion_engine
from src.serving.security import rate_limiter

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}

BATCH = pd.DataFrame({
    "household_key": [1, 2, 3, 42, 77],
    "purchase_probability": [0.95, 0.65, 0.15, 0.50, 0.50],
    "rank": [1, 800, 2400, 1200, 1201],
    "decile": [10, 7, 1, 5, 5],
    "urgency_tier": ["URGENT", "REGULAR", "LOW", "REGULAR", "REGULAR"],
    "marketing_action": ["VIP_COUPON_HALO_UPSELL", "CYCLE_ACCELERATION_NUDGE", "WINBACK_OR_HOLDOUT",
                         "STANDARD_ENGAGEMENT", "STANDARD_ENGAGEMENT"],
})


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(settings, "SERVING_API_KEYS", API_KEY)
    monkeypatch.setattr(settings, "ENV", "development")
    store = fusion_engine.batch_store
    fusion_engine.speed_processor.reset_intraday_deltas()
    store.clear()
    rate_limiter.reset()
    yield
    fusion_engine.speed_processor.reset_intraday_deltas()
    store.clear()
    rate_limiter.reset()
    fusion_engine.router.set_traffic_split(0.10, enabled=False)
    fusion_engine.router.challenger = None


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def with_batch():
    fusion_engine.set_batch_cache(BATCH)


def predict(client, hh):
    resp = client.post("/v1/predict/propensity", json={"household_key": hh})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------ health / ready
def test_health_reports_model_and_batch_view(client, with_batch):
    data = client.get("/health").json()
    assert data["status"] == "HEALTHY"
    assert data["model_target"] == "purchase" and data["horizon_days"] == 14
    assert data["feature_count"] == 28
    assert data["batch_view"]["champion_households"] == len(BATCH)


def test_ready_requires_batch_scores(client):
    assert client.get("/ready").status_code == 503
    fusion_engine.set_batch_cache(BATCH)
    assert client.get("/ready").status_code == 200


# --------------------------------------------------------------- predictions
def test_predict_serves_batch_view(client, with_batch):
    data = predict(client, 1)
    assert data["score_source"] == "batch"
    assert data["unified_propensity"] == pytest.approx(0.95)
    assert data["decile"] == 10 and data["model_variant"] == "champion"


def test_unknown_household_gets_explicit_cold_start(client, with_batch):
    data = predict(client, 9999)
    assert data["score_source"] == "cold_start"
    assert data["decile"] is None and data["marketing_action"] == "COLD_START_DEFAULT"
    assert data["batch_baseline_propensity"] == pytest.approx(BATCH["purchase_probability"].mean(), abs=1e-4)


def test_cold_start_without_any_batch_uses_champion_base_rate(client):
    data = predict(client, 5)
    assert data["score_source"] == "cold_start"
    expected = fusion_engine.predictor.bundle["metrics"]["base_rate"]
    assert data["batch_baseline_propensity"] == pytest.approx(expected, abs=1e-4)


def test_invalid_household_rejected(client):
    assert client.post("/v1/predict/propensity", json={"household_key": 0}).status_code == 422


def test_checkout_event_dampens_and_coupon_boosts(client, with_batch):
    resp = client.post("/v1/speed/event", headers=AUTH,
                       json={"household_key": 42, "event_type": "pos_checkout", "sales_value": 85.5})
    assert resp.status_code == 202
    assert resp.json()["updated_delta"]["intraday_trips_detected"] == 1
    data = predict(client, 42)
    assert data["unified_propensity"] == pytest.approx(0.50 * settings.FUSION_PURCHASE_DAMPENING)
    assert data["marketing_action"] == "POST_PURCHASE_COOLDOWN"

    client.post("/v1/speed/event", headers=AUTH, json={"household_key": 77, "event_type": "coupon_clip"})
    data = predict(client, 77)
    assert data["unified_propensity"] == pytest.approx(0.50 + settings.FUSION_COUPON_BOOST_PER_CLIP)
    assert "coupon_clipping_intent_surge" in data["applied_fusion_rules"]


def test_fusion_coefficients_come_from_settings(client, with_batch, monkeypatch):
    monkeypatch.setattr(settings, "FUSION_COUPON_BOOST_PER_CLIP", 0.30)
    client.post("/v1/speed/event", headers=AUTH, json={"household_key": 77, "event_type": "coupon_clip"})
    assert predict(client, 77)["unified_propensity"] == pytest.approx(0.80)


def test_duplicate_event_is_applied_once(client, with_batch):
    event = {"household_key": 77, "event_type": "coupon_clip", "event_id": "evt-1"}
    first = client.post("/v1/speed/event", headers=AUTH, json=event).json()
    second = client.post("/v1/speed/event", headers=AUTH, json=event).json()
    assert first["event_id"] == second["event_id"] == "evt-1"
    assert second["updated_delta"]["intraday_coupons_clipped"] == 1
    assert second["updated_delta"].get("duplicate") is True


def test_unknown_event_type_rejected(client):
    resp = client.post("/v1/speed/event", headers=AUTH, json={"household_key": 1, "event_type": "teleport"})
    assert resp.status_code == 422


# ----------------------------------------------------------------------- auth
@pytest.mark.parametrize("method,path,body", [
    ("post", "/v1/speed/event", {"household_key": 1, "event_type": "app_open"}),
    ("post", "/v1/canary/config", {"challenger_ratio": 0.2}),
    ("get", "/v1/batch/scores", None),
    ("post", "/v1/models/reload", None),
])
def test_protected_endpoints_require_api_key(client, method, path, body):
    call = getattr(client, method)
    kwargs = {"json": body} if body is not None else {}
    assert call(path, **kwargs).status_code == 401
    assert call(path, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401
    assert call(path, headers=AUTH, **kwargs).status_code in (200, 202)


def test_production_without_keys_fails_closed(client, monkeypatch):
    monkeypatch.setattr(settings, "SERVING_API_KEYS", "")
    monkeypatch.setattr(settings, "ENV", "production")
    assert client.post("/v1/models/reload").status_code == 503


def test_development_without_keys_is_open(client, monkeypatch):
    monkeypatch.setattr(settings, "SERVING_API_KEYS", "")
    assert client.post("/v1/models/reload").status_code == 200


def test_rate_limit(client, with_batch, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 3)
    codes = [client.post("/v1/predict/propensity", json={"household_key": 1}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


# ------------------------------------------------------------------ batch export
def test_batch_scores_endpoint(client, with_batch):
    data = client.get("/v1/batch/scores", headers=AUTH).json()
    assert data["total_available"] == len(BATCH) and data["returned"] == len(BATCH)
    top = client.get("/v1/batch/scores?decile=10", headers=AUTH).json()
    assert top["returned"] == 1 and top["scores"][0]["household_key"] == 1


def test_metrics_and_model_info(client, with_batch):
    metrics = client.get("/v1/metrics").json()
    assert metrics["cached_households"] == len(BATCH)
    info = client.get("/v1/models/champion").json()
    assert info["target"] == "purchase" and info["feature_count"] == 28
    reload_ = client.post("/v1/models/reload", headers=AUTH).json()
    assert reload_["status"] == "RELOADED"


# ---------------------------------------------------------------------- canary
def test_challenger_traffic_gets_challenger_batch_view(client, with_batch):
    challenger_batch = BATCH.assign(purchase_probability=0.33)
    fusion_engine.set_batch_cache(challenger_batch, variant="challenger")
    fusion_engine.router.challenger = fusion_engine.predictor  # same contract; only routing matters here
    fusion_engine.router.set_traffic_split(1.0, enabled=True)
    data = predict(client, 1)
    assert data["routed_variant"] == "challenger" and data["model_variant"] == "challenger"
    assert data["batch_baseline_propensity"] == pytest.approx(0.33)


def test_challenger_without_batch_falls_back_to_champion_view(client, with_batch):
    fusion_engine.router.challenger = fusion_engine.predictor
    fusion_engine.router.set_traffic_split(1.0, enabled=True)
    data = predict(client, 1)
    assert data["routed_variant"] == "challenger" and data["model_variant"] == "champion"
    assert data["batch_baseline_propensity"] == pytest.approx(0.95)
