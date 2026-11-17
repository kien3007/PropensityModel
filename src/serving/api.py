"""
FastAPI Microservice for Enterprise Lambda Serving Gateway.
Exposes REST endpoints for real-time propensity inference, intraday event ingestion,
and precomputed batch scores.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.serving.fusion_engine import LambdaFusionEngine

logger = logging.getLogger("ServingAPI")

app = FastAPI(
    title="Household Purchase Propensity Platform",
    description="Enterprise Lambda Architecture Serving API for Retail Propensity Scoring",
    version="3.0.0",
)

# Global Fusion Engine singleton
fusion_engine = LambdaFusionEngine()


# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    household_key: int = Field(..., ge=1, le=2500, description="Household ID (1 to 2500)")


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
    try:
        return fusion_engine.get_unified_propensity(req.household_key)
    except Exception as e:
        logger.error(f"Prediction failed for household {req.household_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/speed/event", status_code=status.HTTP_202_ACCEPTED)
def ingest_streaming_event(event: StreamingEvent) -> Dict[str, Any]:
    """
    Ingests an intraday event into the Speed Layer (POS checkout, coupon clip, app open).
    Updates real-time delta state in Redis.
    """
    try:
        delta = fusion_engine.speed_processor.process_event(event.model_dump())
        return {
            "status": "ACCEPTED",
            "household_key": event.household_key,
            "event_type": event.event_type,
            "updated_delta": delta,
        }
    except Exception as e:
        logger.error(f"Event ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        return {
            "status": "RELOADED",
            "active_bundle": pred.bundle_name,
            "feature_count": len(pred.features),
        }
    except Exception as e:
        logger.error(f"Failed to reload champion model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
