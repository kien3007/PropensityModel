"""
FastAPI serving gateway: answers from the precomputed weekly batch view.

Open:      GET /health, GET /ready, POST /v1/predict/propensity, GET /v1/canary/config,
           GET /metrics, GET /v1/metrics, GET /v1/models/champion
API key:   POST /v1/canary/config, GET /v1/batch/scores (bulk export), POST /v1/models/reload
Rate limit (per client and route): predict.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.monitoring.metrics import (
    PROPENSITY_REQUESTS_TOTAL,
    generate_prometheus_payload,
    record_model_routing,
    record_prediction,
    refresh_monitoring_gauges,
    update_model_info_gauge,
)
from src.monitoring.state import MonitoringState
from src.serving.scoring_service import PropensityScoringService
from src.serving.security import rate_limit, require_api_key

logger = logging.getLogger("ServingAPI")

app = FastAPI(
    title="Household Purchase Propensity Platform",
    description="Household purchase propensity scores served from the weekly batch view",
    version="4.0.0",
)

# Global singletons. With STRICT_INFRA=true these raise at import if Redis is down.
scoring_service = PropensityScoringService()
scoring_service.batch_store.bootstrap_from_outputs(settings.OUTPUT_DIR)
monitoring_state = MonitoringState()
update_model_info_gauge(scoring_service.predictor.bundle_name, len(scoring_service.predictor.features))

PROTECTED = [Depends(require_api_key)]


# --- Schemas ---
class PredictRequest(BaseModel):
    household_key: int = Field(..., ge=1, description="Household ID; unknown households get a cold-start answer")


class CanaryConfigRequest(BaseModel):
    challenger_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of traffic routed to the challenger")
    enabled: Optional[bool] = True
    challenger_model_path: Optional[str] = None


class PropensityResponse(BaseModel):
    household_key: int
    propensity: float
    decile: Optional[int]
    urgency_tier: str
    marketing_action: str
    model_variant: str = "champion"
    routed_variant: str = "champion"
    score_source: str
    batch_snapshot_day: Optional[int] = None


# --- Endpoints ---
def _batch_status() -> Dict[str, Any]:
    store = scoring_service.batch_store
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
    pred = scoring_service.predictor
    return {
        "status": "HEALTHY",
        "service": "propensity-serving-api",
        "environment": settings.ENV,
        "model_bundle": pred.bundle_name,
        "model_target": pred.target,
        "horizon_days": pred.horizon_days,
        "feature_count": len(pred.features),
        "batch_view": _batch_status(),
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
async def predict_propensity(req: PredictRequest) -> Dict[str, Any]:
    """Propensity from the batch view of the routed variant (cold start when unknown)."""
    # async on purpose: the work is one in-memory lookup or one or two sub-millisecond Redis reads,
    # cheaper than handing every request to the threadpool (the dominant cost when it was sync).
    start_t = time.perf_counter()
    try:
        result = scoring_service.get_propensity(req.household_key)
    except Exception as e:
        PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/predict/propensity", status="500").inc()
        logger.error(f"Prediction failed for household {req.household_key}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    record_prediction(
        latency_seconds=time.perf_counter() - start_t,
        score=result["propensity"],
        decile=result["decile"] if result["decile"] is not None else 0,
        urgency_tier=result["urgency_tier"],
    )
    record_model_routing(result["model_variant"])
    PROPENSITY_REQUESTS_TOTAL.labels(endpoint="/v1/predict/propensity", status="200").inc()
    return result


@app.get("/v1/canary/config")
def get_canary_config() -> Dict[str, Any]:
    status_ = scoring_service.router.get_status()
    status_["challenger"]["batch_households"] = scoring_service.batch_store.count("challenger")
    return status_


@app.post("/v1/canary/config", dependencies=PROTECTED)
def set_canary_config(config: CanaryConfigRequest) -> Dict[str, Any]:
    """Updates the canary split; optionally loads a challenger bundle (must match the champion's target)."""
    if config.challenger_model_path:
        loaded = scoring_service.router.load_challenger(config.challenger_model_path)
        if not loaded:
            raise HTTPException(status_code=400, detail=f"Failed to load challenger model from {config.challenger_model_path}")
    result = scoring_service.router.set_traffic_split(challenger_ratio=config.challenger_ratio, enabled=config.enabled)
    if result["canary_enabled"] and scoring_service.batch_store.count("challenger") == 0:
        result["warning"] = "No challenger batch scores published yet: challenger traffic is served the champion view."
    return result


@app.get("/v1/batch/scores", dependencies=PROTECTED)
def get_batch_scores(decile: Optional[int] = None, limit: int = 50, variant: str = "champion") -> Dict[str, Any]:
    """Bulk export of precomputed batch views (e.g. decile=10 for the top targets)."""
    store = scoring_service.batch_store
    if variant not in ("champion", "challenger"):
        raise HTTPException(status_code=422, detail="variant must be 'champion' or 'challenger'")
    records = store.records(variant, decile=decile, limit=max(0, min(limit, 5000)))
    return {"variant": variant, "total_available": store.count(variant), "returned": len(records),
            "metadata": store.metadata(variant), "scores": records}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    refresh_monitoring_gauges(monitoring_state, scoring_service.batch_store)
    return Response(content=generate_prometheus_payload(), media_type="text/plain")


@app.get("/v1/metrics")
def get_metrics() -> Dict[str, Any]:
    return {
        "cached_households": scoring_service.batch_store.count("champion"),
        "batch_view": _batch_status(),
        "model_features": scoring_service.predictor.features,
    }


@app.get("/v1/models/champion")
def get_champion_model_info() -> Dict[str, Any]:
    pred = scoring_service.predictor
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
        scoring_service.reload_predictor()
    except Exception as e:
        logger.error(f"Failed to reload champion model: {e}")
        raise HTTPException(status_code=500, detail="Champion reload failed") from e
    pred = scoring_service.predictor
    update_model_info_gauge(pred.bundle_name, len(pred.features))
    return {"status": "RELOADED", "active_bundle": pred.bundle_name, "feature_count": len(pred.features)}
