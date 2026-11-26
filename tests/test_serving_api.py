"""
Serving API tests: batch-view serving, cold start, API-key auth, rate limiting, readiness and
canary variant serving.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config.settings import settings
from src.serving.api import app, scoring_service
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
    store = scoring_service.batch_store
    store.clear()
    rate_limiter.reset()
    yield
    store.clear()
    rate_limiter.reset()
    scoring_service.router.set_traffic_split(0.10, enabled=False)
    scoring_service.router.challenger = None


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def with_batch():
    scoring_service.set_batch_cache(BATCH)


def predict(client, hh):
    resp = client.post("/v1/predict/propensity", json={"household_key": hh})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------ health / ready
def test_health_reports_model_and_batch_view(client, with_batch):
    data = client.get("/health").json()
    assert data["status"] == "HEALTHY"
    assert data["model_target"] == "purchase" and data["horizon_days"] == 14
    assert data["feature_count"] == len(scoring_service.predictor.features)
    assert data["batch_view"]["champion_households"] == len(BATCH)


def test_ready_requires_batch_scores(client):
    assert client.get("/ready").status_code == 503
    scoring_service.set_batch_cache(BATCH)
    assert client.get("/ready").status_code == 200


# --------------------------------------------------------------- predictions
def test_predict_serves_batch_view(client, with_batch):
    data = predict(client, 1)
    assert data["score_source"] == "batch" and data["urgency_tier"] == "URGENT"
    assert set(data) == {"household_key", "propensity", "decile", "urgency_tier", "marketing_action",
                         "model_variant", "routed_variant", "score_source", "batch_snapshot_day"}
    assert data["propensity"] == pytest.approx(0.95)
    assert data["decile"] == 10 and data["model_variant"] == "champion"


def test_unknown_household_gets_explicit_cold_start(client, with_batch):
    data = predict(client, 9999)
    assert data["score_source"] == "cold_start"
    assert data["decile"] is None and data["marketing_action"] == "COLD_START_DEFAULT"
    assert data["propensity"] == pytest.approx(BATCH["purchase_probability"].mean(), abs=1e-4)


def test_cold_start_without_any_batch_uses_champion_base_rate(client):
    data = predict(client, 5)
    assert data["score_source"] == "cold_start"
    expected = scoring_service.predictor.bundle["metrics"]["base_rate"]
    assert data["propensity"] == pytest.approx(expected, abs=1e-4)


def test_invalid_household_rejected(client):
    assert client.post("/v1/predict/propensity", json={"household_key": 0}).status_code == 422


def test_streaming_endpoint_is_gone(client):
    resp = client.post("/v1/speed/event", headers=AUTH, json={"household_key": 1, "event_type": "pos_checkout"})
    assert resp.status_code in (404, 405)


# ----------------------------------------------------------------------- auth
@pytest.mark.parametrize("method,path,body", [
    ("post", "/v1/canary/config", {"challenger_ratio": 0.2}),
    ("get", "/v1/batch/scores", None),
    ("post", "/v1/models/reload", None),
])
def test_protected_endpoints_require_api_key(client, method, path, body):
    call = getattr(client, method)
    kwargs = {"json": body} if body is not None else {}
    assert call(path, **kwargs).status_code == 401
    assert call(path, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401
    assert call(path, headers=AUTH, **kwargs).status_code == 200


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
    assert info["target"] == "purchase" and info["feature_set_version"] == "multidomain_v2" and info["feature_count"] == 27
    reload_ = client.post("/v1/models/reload", headers=AUTH).json()
    assert reload_["status"] == "RELOADED"


# ---------------------------------------------------------------------- canary
def test_challenger_traffic_gets_challenger_batch_view(client, with_batch):
    challenger_batch = BATCH.assign(purchase_probability=0.33)
    scoring_service.set_batch_cache(challenger_batch, variant="challenger")
    scoring_service.router.challenger = scoring_service.predictor  # same contract; only routing matters here
    scoring_service.router.set_traffic_split(1.0, enabled=True)
    data = predict(client, 1)
    assert data["routed_variant"] == "challenger" and data["model_variant"] == "challenger"
    assert data["propensity"] == pytest.approx(0.33)


def test_challenger_without_batch_falls_back_to_champion_view(client, with_batch):
    scoring_service.router.challenger = scoring_service.predictor
    scoring_service.router.set_traffic_split(1.0, enabled=True)
    data = predict(client, 1)
    assert data["routed_variant"] == "challenger" and data["model_variant"] == "champion"
    assert data["propensity"] == pytest.approx(0.95)
