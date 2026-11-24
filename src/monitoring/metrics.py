"""
Prometheus Metrics Instrumentation Module for Propensity Serving Gateway.
Tracks Request Latency (P50/P95/P99), RPS, Deciles, Urgency Tiers, Fusion Rules, and Drift PSI.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from prometheus_client import (
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

PROPENSITY_MODEL_ROUTING_TOTAL = Counter(
    "propensity_model_routing_total",
    "Total predictions broken down by A/B Canary model variant",
    ["variant"],
)


# Model quality & monitoring (refreshed from MonitoringState when /metrics is scraped)
PROPENSITY_REALIZED_AUC = Gauge(
    "propensity_realized_roc_auc", "Realized out-of-time ROC-AUC from the latest ground-truth feedback", ["target"])
PROPENSITY_EXPECTED_AUC = Gauge(
    "propensity_expected_roc_auc", "ROC-AUC the active champion was accepted with (OOT test)", ["target"])
PROPENSITY_REALIZED_AUC_DROP = Gauge(
    "propensity_realized_auc_drop", "Expected minus realized ROC-AUC (positive = degradation)", ["target"])
PROPENSITY_FEEDBACK_CUTOFF_DAY = Gauge(
    "propensity_feedback_cutoff_day", "Cutoff DAY of the latest ground-truth evaluation", ["target"])
PROPENSITY_DRIFT_STATUS = Gauge(
    "propensity_drift_status", "Latest drift audit status (0 = stable, 1 = warning, 2 = critical)", ["target"])
PROPENSITY_DRIFT_RATIO = Gauge(
    "propensity_drift_ratio", "Share of features in significant drift in the latest audit", ["target"])
PROPENSITY_PREDICTION_DRIFT_PSI = Gauge(
    "propensity_prediction_drift_psi", "PSI of the scored probability distribution vs the champion reference",
    ["target"])
PROPENSITY_BATCH_VIEW_HOUSEHOLDS = Gauge(
    "propensity_batch_view_households", "Households with a published batch view", ["variant"])
PROPENSITY_BATCH_SNAPSHOT_DAY = Gauge(
    "propensity_batch_snapshot_day", "Snapshot DAY of the served champion batch view")

PROPENSITY_CONSUMER_LAG = Gauge(
    "propensity_speed_consumer_lag", "Kafka consumer lag (messages) on the intraday events topic")
PROPENSITY_CONSUMED_EVENTS_TOTAL = Counter(
    "propensity_speed_consumed_events_total", "Events applied by the Kafka consumer worker", ["outcome"])

PROPENSITY_FS_SNAPSHOT_DAY = Gauge(
    "propensity_feature_store_snapshot_day", "DAY of the latest profiled feature store snapshot")
PROPENSITY_FS_DRIFT_SHARE = Gauge(
    "propensity_feature_store_drift_share",
    "Share of features that drifted between the latest snapshot and the previous one (Evidently)")
PROPENSITY_FS_FEATURE_DRIFT = Gauge(
    "propensity_feature_store_feature_drift_score",
    "Drift score per feature vs the previous snapshot (normed Wasserstein / Jensen-Shannon)", ["feature_name"])
PROPENSITY_FS_OUT_OF_RANGE = Gauge(
    "propensity_feature_store_out_of_range_share",
    "Share of values outside the previous snapshot's [min, max], per feature", ["feature_name"])
PROPENSITY_FS_ALMOST_CONSTANT = Gauge(
    "propensity_feature_store_almost_constant_columns", "Almost-constant feature columns in the latest snapshot")
PROPENSITY_FS_QUALITY_PASSED = Gauge(
    "propensity_feature_store_quality_passed", "1 if the latest snapshot passed the quality gate, else 0")
PROPENSITY_FS_QUALITY_VIOLATIONS = Gauge(
    "propensity_feature_store_quality_violations", "Quality gate violations of the latest snapshot")
PROPENSITY_FS_QUALITY_DAY = Gauge(
    "propensity_feature_store_quality_cutoff_day", "DAY of the latest snapshot checked by the quality gate")

DRIFT_STATUS_CODES = {"STABLE": 0, "WARNING": 1, "CRITICAL_DRIFT": 2}


def refresh_monitoring_gauges(state, batch_store=None) -> None:
    """Copies the latest cross-process monitoring reports into Prometheus gauges."""
    for target in ("purchase", "lapse"):
        fb = state.latest(f"feedback_{target}")
        if fb:
            PROPENSITY_REALIZED_AUC.labels(target=target).set(fb["realized_roc_auc"])
            PROPENSITY_EXPECTED_AUC.labels(target=target).set(fb["expected_roc_auc"])
            PROPENSITY_REALIZED_AUC_DROP.labels(target=target).set(fb["auc_drop"])
            PROPENSITY_FEEDBACK_CUTOFF_DAY.labels(target=target).set(fb["cutoff_day"])
        drift = state.latest(f"drift_{target}")
        if drift:
            PROPENSITY_DRIFT_STATUS.labels(target=target).set(DRIFT_STATUS_CODES.get(drift["overall_status"], 0))
            PROPENSITY_DRIFT_RATIO.labels(target=target).set(drift["feature_drift_summary"]["drift_ratio"])
            if drift.get("prediction_drift"):
                PROPENSITY_PREDICTION_DRIFT_PSI.labels(target=target).set(drift["prediction_drift"]["psi"])
            if target == "purchase":
                record_drift_metrics({f: d["psi"] for f, d in drift.get("feature_details", {}).items()})
    refresh_feature_store_gauges(state)
    if batch_store is not None:
        for variant in ("champion", "challenger"):
            PROPENSITY_BATCH_VIEW_HOUSEHOLDS.labels(variant=variant).set(batch_store.count(variant))
        meta = batch_store.metadata("champion") or {}
        if meta.get("snapshot_day") is not None:
            PROPENSITY_BATCH_SNAPSHOT_DAY.set(meta["snapshot_day"])


def refresh_feature_store_gauges(state) -> None:
    quality = state.latest("feature_store_quality")
    if quality:
        PROPENSITY_FS_QUALITY_PASSED.set(1 if quality["passed"] else 0)
        PROPENSITY_FS_QUALITY_VIOLATIONS.set(len(quality.get("violations") or []))
        PROPENSITY_FS_QUALITY_DAY.set(quality["cutoff_day"])
    profile = state.latest("feature_store_profile")
    if not profile:
        return
    PROPENSITY_FS_SNAPSHOT_DAY.set(profile["cutoff_day"])
    PROPENSITY_FS_ALMOST_CONSTANT.set(profile.get("almost_constant_columns") or 0)
    drift = profile.get("drift")
    if drift:
        PROPENSITY_FS_DRIFT_SHARE.set(drift["drifted_share"] or 0)
        for feature, col in drift["columns"].items():
            PROPENSITY_FS_FEATURE_DRIFT.labels(feature_name=feature).set(col["score"])
    for feature, oor in (profile.get("out_of_range") or {}).items():
        PROPENSITY_FS_OUT_OF_RANGE.labels(feature_name=feature).set(oor["share"])


def record_model_routing(variant: str) -> None:
    """Records routing to champion or challenger."""
    PROPENSITY_MODEL_ROUTING_TOTAL.labels(variant=variant).inc()


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
