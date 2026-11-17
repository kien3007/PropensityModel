"""
FastAPI Microservice for Enterprise Lambda Serving Gateway.
Exposes REST endpoints for real-time propensity inference, intraday event ingestion,
and precomputed batch scores.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.monitoring.metrics import (
    PROPENSITY_REQUESTS_TOTAL,
    generate_prometheus_payload,
    record_model_routing,
    record_prediction,
    record_speed_event,
    update_model_info_gauge,
)
from src.serving.fusion_engine import LambdaFusionEngine
from src.speed_layer.kafka_bus import KafkaEventProducer

logger = logging.getLogger("ServingAPI")

app = FastAPI(
    title="Household Purchase Propensity Platform",
    description="Enterprise Lambda Architecture Serving API for Retail Propensity Scoring",
    version="3.0.0",
)

# Global Singletons
fusion_engine = LambdaFusionEngine()
kafka_producer = KafkaEventProducer(fallback_processor=fusion_engine.speed_processor)
update_model_info_gauge(fusion_engine.predictor.bundle_name, len(fusion_engine.predictor.features))


# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    household_key: int = Field(..., ge=1, le=2500, description="Household ID (1 to 2500)")


class CanaryConfigRequest(BaseModel):
    challenger_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of traffic (0.0 to 1.0) to route to Challenger")
    enabled: Optional[bool] = True
    challenger_model_path: Optional[str] = None


class StreamingEvent(BaseModel):
    household_key: int = Field(..., ge=1, le=2500)
    event_type: str = Field(..., description="Event type: 'pos_checkout', 'coupon_clip', 'app_open'")
    sales_value: Optional[float] = Field(0.0, ge=0.0)
    coupon_id: Optional[int] = None


class PropensityResponse(BaseModel):
    household_key: int
    unified_propensity: float
    batch_baseline_propensity: float
    decile: int
    urgency_tier: str
    marketing_action: str
    intraday_delta: Dict[str, Any]
    applied_fusion_rules: List[str]
    model_variant: Optional[str] = "champion"


# --- Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Health & Readiness probe for Kubernetes / Docker."""
    return {
        "status": "HEALTHY",
        "service": "propensity-serving-api",
        "environment": settings.ENV,
        "model_bundle": fusion_engine.predictor.bundle_name,
        "feature_count": len(fusion_engine.predictor.features),
    }


@app.post("/v1/predict/propensity", response_model=PropensityResponse)
def predict_propensity(req: PredictRequest) -> Dict[str, Any]:
    """
    Returns real-time unified propensity score for a household.
    Fuses batch baseline with live intraday speed deltas.
    """
    start_t = time.perf_counter()
    try:
        result = fusion_engine.get_unified_propensity(req.household_key)
        latency = time.perf_counter() - start_t
        variant = result.get("model_variant", "champion")
        record_prediction(
            latency_seconds=latency,
            score=result["unified_propensity"],
            decile=result["decile"],
            urgency_tier=result["urgency_tier"],
            rules=result.get("applied_fusion_rules", []),
        )
        record_model_routing(variant)
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/predict/propensity", status="200").inc()
        return result
    except Exception as e:
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/predict/propensity", status="500").inc()
        logger.error(f"Prediction failed for household {req.household_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/speed/event", status_code=status.HTTP_202_ACCEPTED)
def ingest_streaming_event(event: StreamingEvent) -> Dict[str, Any]:
    """
    Ingests an intraday event into the Speed Layer (POS checkout, coupon clip, app open).
    Dispatches to Kafka Event Bus with zero-downtime direct Redis fallback.
    """
    try:
        dispatch = kafka_producer.publish_event(event.model_dump())
        record_speed_event(event.event_type)
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/speed/event", status="202").inc()
        return {
            "status": "ACCEPTED",
            "household_key": event.household_key,
            "event_type": event.event_type,
            "dispatch": dispatch,
            "updated_delta": dispatch.get("updated_delta") or fusion_engine.speed_processor.get_delta(event.household_key),
        }
    except Exception as e:
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/speed/event", status="500").inc()
        logger.error(f"Event ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/canary/config")
def get_canary_config() -> Dict[str, Any]:
    """Returns current A/B Canary routing configuration and split percentages."""
    return fusion_engine.router.get_status()


@app.post("/v1/canary/config")
def set_canary_config(config: CanaryConfigRequest) -> Dict[str, Any]:
    """Dynamically updates A/B Canary traffic ratio and Challenger model path."""
    if config.challenger_model_path:
        loaded = fusion_engine.router.load_challenger(config.challenger_model_path)
        if not loaded:
            raise HTTPException(status_code=400, detail=f"Failed to load challenger model from {config.challenger_model_path}")

    return fusion_engine.router.set_traffic_split(
        challenger_ratio=config.challenger_ratio,
        enabled=config.enabled,
    )


@app.get("/v1/batch/scores")
def get_batch_scores(decile: Optional[int] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Retrieves precomputed batch scores cached in Serving Layer.
    Can be filtered by decile (e.g. decile=10 for top VIP targets).
    """
    cache = fusion_engine._batch_cache
    if not cache:
        return {"total_cached": 0, "scores": []}

    records = list(cache.values())
    if decile is not None:
        records = [r for r in records if r["decile"] == decile]

    records = sorted(records, key=lambda x: x.get("rank", 9999))[:limit]
    return {
        "total_available": len(cache),
        "returned": len(records),
        "scores": records,
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Exposes real-time Prometheus scrapable metrics."""
    return Response(content=generate_prometheus_payload(), media_type="text/plain")


@app.get("/v1/metrics")
def get_metrics() -> Dict[str, Any]:
    """Exposes system metrics and summary stats."""
    cache = fusion_engine._batch_cache
    return {
        "cached_households": len(cache),
        "active_speed_deltas": len(fusion_engine.speed_processor._memory_fallback),
        "model_features": fusion_engine.predictor.features,
    }


@app.get("/v1/models/champion")
def get_champion_model_info() -> Dict[str, Any]:
    """Returns active Champion model metadata and feature contract."""
    pred = fusion_engine.predictor
    return {
        "bundle_name": pred.bundle_name,
        "feature_count": len(pred.features),
        "features": pred.features,
        "has_calibrator": pred.calibrator is not None,
        "epsilon": pred.epsilon,
    }


@app.post("/v1/models/reload")
def reload_champion_model() -> Dict[str, Any]:
    """
    Zero-Downtime Hot-Reload:
    Reloads the latest promoted champion model bundle into memory without restarting the API service.
    """
    try:
        fusion_engine.reload_predictor()
        pred = fusion_engine.predictor
        update_model_info_gauge(pred.bundle_name, len(pred.features))
        return {
            "status": "RELOADED",
            "active_bundle": pred.bundle_name,
            "feature_count": len(pred.features),
        }
    except Exception as e:
        logger.error(f"Failed to reload champion model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
