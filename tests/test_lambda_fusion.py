"""
Unit tests for the Lambda Serving Fusion Engine.
Verifies fusion of batch baseline with real-time speed layer deltas.
"""
import pytest
import pandas as pd
from src.serving.fusion_engine import LambdaFusionEngine
from src.speed_layer.delta_stream import SpeedLayerProcessor


def test_fusion_engine_rules():
    speed_proc = SpeedLayerProcessor()
    speed_proc.reset_intraday_deltas()

    engine = LambdaFusionEngine(speed_processor=speed_proc)

    # Mock batch scores
    batch_df = pd.DataFrame({
        "household_key": [10, 20, 30],
        "purchase_probability": [0.80, 0.50, 0.20],
        "rank": [200, 1200, 2200],
        "decile": [9, 5, 2],
        "urgency_tier": ["URGENT", "REGULAR", "LOW"],
        "marketing_action": ["VIP_COUPON_HALO_UPSELL", "STANDARD_ENGAGEMENT", "WINBACK_OR_HOLDOUT"],
    })
    engine.set_batch_cache(batch_df)

    # Test baseline without deltas
    res_base = engine.get_unified_propensity(10)
    assert res_base["unified_propensity"] == 0.80
    assert res_base["urgency_tier"] == "URGENT"

    # Test Rule 1: Intraday purchase dampening
    speed_proc.process_event({
        "household_key": 10,
        "event_type": "pos_checkout",
        "sales_value": 75.0,
    })
    res_shopped = engine.get_unified_propensity(10)
    assert res_shopped["unified_propensity"] == pytest.approx(0.80 * 0.25)
    assert res_shopped["urgency_tier"] == "LOW"
    assert "intraday_purchase_dampening" in res_shopped["applied_fusion_rules"]

    # Test Rule 2: Coupon clip intent surge
    speed_proc.process_event({
        "household_key": 20,
        "event_type": "coupon_clip",
        "coupon_id": 999,
    })
    res_coupon = engine.get_unified_propensity(20)
    assert res_coupon["unified_propensity"] == pytest.approx(0.50 + 0.15)
    assert res_coupon["urgency_tier"] == "URGENT"
    assert "coupon_clipping_intent_surge" in res_coupon["applied_fusion_rules"]
