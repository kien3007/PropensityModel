"""
Unit tests for A/B Testing, Canary Router, and Dynamic Traffic Split.
"""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.models.predictor import PropensityPredictor
from src.serving.api import app, fusion_engine
from src.serving.canary_router import CanaryRouter


def test_canary_router_disabled_by_default():
    router = CanaryRouter(enabled=False)
    predictor, variant = router.route(household_key=100)
    assert variant == "champion"
    assert predictor == router.champion


def test_canary_router_deterministic_split():
    champion = PropensityPredictor()
    challenger = MagicMock(spec=PropensityPredictor)
    challenger.bundle_name = "test_challenger"
    challenger.features = ["f1", "f2"]

    router = CanaryRouter(
        champion_predictor=champion,
        challenger_traffic_ratio=0.30,  # 30% to Challenger
        enabled=True,
    )
    router.challenger = challenger

    # 1. Determinism test: same household always yields same variant
    _, v1 = router.route(household_key=42)
    _, v2 = router.route(household_key=42)
    assert v1 == v2

    # 2. Distribution test across 1000 households
    variants = [router.route(hh)[1] for hh in range(1, 1001)]
    challenger_count = sum(1 for v in variants if v == "challenger")
    champion_count = sum(1 for v in variants if v == "champion")

    # With 30% ratio, challenger should receive roughly 250-350 households
    assert 200 <= challenger_count <= 400
    assert champion_count + challenger_count == 1000


def test_canary_api_endpoints():
    client = TestClient(app)

    # 1. Get initial status
    status_res = client.get("/v1/canary/config")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert "traffic_split" in status_data
    assert "champion_percentage" in status_data["traffic_split"]

    # 2. Update traffic ratio to 25%
    update_res = client.post("/v1/canary/config", json={"challenger_ratio": 0.25, "enabled": True})
    assert update_res.status_code == 200
    updated_data = update_res.json()
    assert updated_data["traffic_split"]["challenger_percentage"] in [0, 25]

    # 3. Predict propensity returns model_variant
    pred_res = client.post("/v1/predict/propensity", json={"household_key": 100})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "model_variant" in pred_data
    assert pred_data["model_variant"] in ["champion", "challenger"]
