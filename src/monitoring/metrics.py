"""
Prometheus Metrics Instrumentation Module for Propensity Serving Gateway.
Tracks Request Latency (P50/P95/P99), RPS, Deciles, Urgency Tiers, Fusion Rules, and Drift PSI.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
)

# Core Metrics
PROPENSITY_REQUESTS_TOTAL = Counter(
    "propensity_requests_total",
    "Total HTTP requests handled by Serving API",
    ["endpoint", "status"],
)

PROPENSITY_PREDICTION_LATENCY = Histogram(
    "propensity_prediction_latency_seconds",
    "Prediction and fusion inference latency in seconds",
    ["endpoint"],
    buckets=(0.001, 0.003, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000),
)

PROPENSITY_UNIFIED_SCORE = Histogram(
    "propensity_unified_score",
    "Distribution of predicted unified propensity probabilities",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

PROPENSITY_DECILE_TOTAL = Counter(
    "propensity_decile_total",
    "Total predictions broken down by Decile tier",
    ["decile"],
)

PROPENSITY_URGENCY_TIER_TOTAL = Counter(
    "propensity_urgency_tier_total",
    "Total predictions broken down by business urgency tier",
    ["tier"],
)

PROPENSITY_FUSION_RULES_TOTAL = Counter(
    "propensity_fusion_rules_total",
    "Number of times each Lambda Serving fusion rule was triggered",
    ["rule_name"],
)

PROPENSITY_SPEED_EVENTS_TOTAL = Counter(
    "propensity_speed_events_total",
    "Total streaming intraday events ingested into Redis Speed Layer",
    ["event_type"],
)

PROPENSITY_FEATURE_DRIFT_PSI = Gauge(
    "propensity_feature_drift_psi",
    "Latest Population Stability Index (PSI) score per feature",
    ["feature_name"],
)

PROPENSITY_ACTIVE_MODEL_INFO = Gauge(
    "propensity_active_model_info",
    "Active Champion Model bundle metadata in Serving RAM",
    ["bundle_name", "feature_count"],
)


def record_prediction(
    latency_seconds: float,
    score: float,
    decile: int,
    urgency_tier: str,
    rules: Optional[List[str]] = None,
) -> None:
    """Records real-time metrics for a single prediction request."""
    PROPENSITY_PREDICTION_LATENCY.labels(endpoint="/v1/predict/propensity").observe(latency_seconds)
    PROPENSITY_UNIFIED_SCORE.observe(score)
    PROPENSITY_DECILE_TOTAL.labels(decile=str(decile)).inc()
    PROPENSITY_URGENCY_TIER_TOTAL.labels(tier=str(urgency_tier)).inc()

    if rules:
        for rule in rules:
            PROPENSITY_FUSION_RULES_TOTAL.labels(rule_name=rule).inc()


def record_speed_event(event_type: str) -> None:
    """Records streaming speed layer event."""
    PROPENSITY_SPEED_EVENTS_TOTAL.labels(event_type=event_type).inc()


def record_drift_metrics(feature_psi_dict: Dict[str, float]) -> None:
    """Updates gauge metrics with latest drift PSI scores per feature."""
    for feature_name, psi_val in feature_psi_dict.items():
        if psi_val is not None:
            PROPENSITY_FEATURE_DRIFT_PSI.labels(feature_name=feature_name).set(float(psi_val))


def update_model_info_gauge(bundle_name: str, feature_count: int) -> None:
    """Updates active champion model metadata gauge."""
    PROPENSITY_ACTIVE_MODEL_INFO.labels(
        bundle_name=bundle_name,
        feature_count=str(feature_count),
    ).set(1)


def generate_prometheus_payload() -> bytes:
    """Serializes all Prometheus metrics into the standard text/plain format."""
    return generate_latest(REGISTRY)
