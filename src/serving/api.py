"""
FastAPI Serving Gateway (Lambda serving layer).

Open:      GET /health, GET /ready, POST /v1/predict/propensity, GET /v1/canary/config,
           GET /metrics, GET /v1/metrics, GET /v1/models/champion
API key:   POST /v1/speed/event, POST /v1/canary/config, GET /v1/batch/scores (bulk export),
           POST /v1/models/reload
Rate limit (per client and route): predict and event ingestion.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from src.common.redis_client import InfrastructureUnavailable
from src.config.settings import settings
from src.monitoring.metrics import (
    PROPENSITY_REQUESTS_TOTAL,
    generate_prometheus_payload,
    record_model_routing,
    record_prediction,
    record_speed_event,
    refresh_monitoring_gauges,
    update_model_info_gauge,
)
from src.monitoring.prediction_log import PredictionLog
from src.monitoring.state import MonitoringState
from src.serving.fusion_engine import LambdaFusionEngine
from src.serving.security import rate_limit, require_api_key
from src.speed_layer.kafka_bus import KafkaEventProducer

logger = logging.getLogger("ServingAPI")

app = FastAPI(
    title="Household Purchase Propensity Platform",
    description="Enterprise Lambda Architecture Serving API for Retail Propensity Scoring",
    version="3.1.0",
)

# Global singletons. With STRICT_INFRA=true these raise at import if Redis/Kafka are down.
fusion_engine = LambdaFusionEngine()
kafka_producer = KafkaEventProducer(fallback_processor=fusion_engine.speed_processor)
fusion_engine.batch_store.bootstrap_from_outputs(settings.OUTPUT_DIR)
monitoring_state = MonitoringState()
prediction_log = PredictionLog()
update_model_info_gauge(fusion_engine.predictor.bundle_name, len(fusion_engine.predictor.features))

PROTECTED = [Depends(require_api_key)]


# --- Schemas ---
class PredictRequest(BaseModel):
    household_key: int = Field(..., ge=1, description="Household ID; unknown households get a cold-start answer")


class CanaryConfigRequest(BaseModel):
    challenger_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of traffic routed to the challenger")
    enabled: Optional[bool] = True
    challenger_model_path: Optional[str] = None


class StreamingEvent(BaseModel):
    household_key: int = Field(..., ge=1)
    event_type: str = Field(..., pattern="^(pos_checkout|coupon_clip|app_open)$")
    sales_value: Optional[float] = Field(0.0, ge=0.0)
    coupon_id: Optional[int] = None
    event_id: Optional[str] = Field(None, max_length=128, description="Idempotency key; generated if omitted")


class PropensityResponse(BaseModel):
    household_key: int
    unified_propensity: float
    batch_baseline_propensity: float
    decile: Optional[int]
    urgency_tier: str
    marketing_action: str
    intraday_delta: Dict[str, Any]
    applied_fusion_rules: List[str]
    model_variant: str = "champion"
    routed_variant: str = "champion"
    score_source: str
    batch_snapshot_day: Optional[int] = None


# --- Endpoints ---
def _batch_status() -> Dict[str, Any]:
    store = fusion_engine.batch_store
    meta = store.metadata("champion") or {}
    return {
        "backend": store.backend,
        "champion_households": store.count("champion"),
        "challenger_households": store.count("challenger"),
        "snapshot_day": meta.get("snapshot_day"),
        "run_id": meta.get("run_id"),
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Liveness: the process is up and a champion bundle is loaded."""
    pred = fusion_engine.predictor
    return {
        "status": "HEALTHY",
        "service": "propensity-serving-api",
        "environment": settings.ENV,
        "model_bundle": pred.bundle_name,
        "model_target": pred.target,
        "horizon_days": pred.horizon_days,
        "feature_count": len(pred.features),
        "batch_view": _batch_status(),
        "speed_layer_backend": fusion_engine.speed_processor.backend,
    }


@app.get("/ready")
def readiness(response: Response) -> Dict[str, Any]:
    """Readiness: 503 until champion batch scores are available (otherwise every answer is cold start)."""
    batch = _batch_status()
    ready = batch["champion_households"] > 0
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "batch_view": batch}


@app.post("/v1/predict/propensity", response_model=PropensityResponse, dependencies=[Depends(rate_limit)])
def predict_propensity(req: PredictRequest) -> Dict[str, Any]:
    """Unified propensity: batch baseline for the routed variant fused with intraday deltas."""
    start_t = time.perf_counter()
    try:
        result = fusion_engine.get_unified_propensity(req.household_key)
    except Exception as e:
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/predict/propensity", status="500").inc()
        logger.error(f"Prediction failed for household {req.household_key}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    record_prediction(
        latency_seconds=time.perf_counter() - start_t,
        score=result["unified_propensity"],
        decile=result["decile"] if result["decile"] is not None else 0,
        urgency_tier=result["urgency_tier"],
        rules=result.get("applied_fusion_rules", []),
    )
    record_model_routing(result["model_variant"])
    prediction_log.append(result)
    PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/predict/propensity", status="200").inc()
    return result


@app.post("/v1/speed/event", status_code=status.HTTP_202_ACCEPTED, dependencies=PROTECTED + [Depends(rate_limit)])
def ingest_streaming_event(event: StreamingEvent) -> Dict[str, Any]:
    """Ingests an intraday event (Kafka, or direct speed-layer update when Kafka is unavailable)."""
    try:
        dispatch = kafka_producer.publish_event(event.model_dump())
    except InfrastructureUnavailable as e:
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/speed/event", status="503").inc()
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/speed/event", status="500").inc()
        logger.error(f"Event ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Event ingestion failed") from e
    record_speed_event(event.event_type)
    PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/speed/event", status="202").inc()
    return {
        "status": "ACCEPTED",
        "household_key": event.household_key,
        "event_type": event.event_type,
        "event_id": dispatch.get("event_id"),
        "dispatch": dispatch,
        "updated_delta": dispatch.get("updated_delta") or fusion_engine.speed_processor.get_delta(event.household_key),
    }


@app.get("/v1/canary/config")
def get_canary_config() -> Dict[str, Any]:
    status_ = fusion_engine.router.get_status()
    status_["challenger"]["batch_households"] = fusion_engine.batch_store.count("challenger")
    return status_


@app.post("/v1/canary/config", dependencies=PROTECTED)
def set_canary_config(config: CanaryConfigRequest) -> Dict[str, Any]:
    """Updates the canary split; optionally loads a challenger bundle (must match the champion's target)."""
    if config.challenger_model_path:
        loaded = fusion_engine.router.load_challenger(config.challenger_model_path)
        if not loaded:
            raise HTTPException(status_code=400, detail=f"Failed to load challenger model from {config.challenger_model_path}")
    result = fusion_engine.router.set_traffic_split(challenger_ratio=config.challenger_ratio, enabled=config.enabled)
    if result["canary_enabled"] and fusion_engine.batch_store.count("challenger") == 0:
        result["warning"] = "No challenger batch scores published yet: challenger traffic is served the champion view."
    return result


@app.get("/v1/batch/scores", dependencies=PROTECTED)
def get_batch_scores(decile: Optional[int] = None, limit: int = 50, variant: str = "champion") -> Dict[str, Any]:
    """Bulk export of precomputed batch views (e.g. decile=10 for the top targets)."""
    store = fusion_engine.batch_store
    if variant not in ("champion", "challenger"):
        raise HTTPException(status_code=422, detail="variant must be 'champion' or 'challenger'")
    records = store.records(variant, decile=decile, limit=max(0, min(limit, 5000)))
    return {"variant": variant, "total_available": store.count(variant), "returned": len(records),
            "metadata": store.metadata(variant), "scores": records}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    refresh_monitoring_gauges(monitoring_state, fusion_engine.batch_store)
    return Response(content=generate_prometheus_payload(), media_type="text/plain")


@app.get("/v1/metrics")
def get_metrics() -> Dict[str, Any]:
    return {
        "cached_households": fusion_engine.batch_store.count("champion"),
        "batch_view": _batch_status(),
        "active_speed_deltas": fusion_engine.speed_processor.active_delta_count(),
        "model_features": fusion_engine.predictor.features,
    }


@app.get("/v1/models/champion")
def get_champion_model_info() -> Dict[str, Any]:
    pred = fusion_engine.predictor
    return {
        "bundle_name": pred.bundle_name,
        "target": pred.target,
        "horizon_days": pred.horizon_days,
        "feature_set_version": pred.feature_set_version,
        "feature_count": len(pred.features),
        "features": pred.features,
        "has_calibrator": pred.calibrator is not None,
        "epsilon": pred.epsilon,
    }


@app.post("/v1/models/reload", dependencies=PROTECTED)
def reload_champion_model() -> Dict[str, Any]:
    """Zero-downtime hot-reload of the champion bundle from the registry."""
    try:
        fusion_engine.reload_predictor()
    except Exception as e:
        logger.error(f"Failed to reload champion model: {e}")
        raise HTTPException(status_code=500, detail="Champion reload failed") from e
    pred = fusion_engine.predictor
    update_model_info_gauge(pred.bundle_name, len(pred.features))
    return {"status": "RELOADED", "active_bundle": pred.bundle_name, "feature_count": len(pred.features)}
