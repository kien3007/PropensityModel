"""
Retraining policy: turns monitoring reports into a retrain decision.

Retrain when
- the drift audit of the latest batch is CRITICAL (>= 20% of features with PSI >= 0.25 or a
  critical shift of the prediction distribution), or
- a target's realized out-of-time AUC fell more than settings.REALIZED_AUC_DROP_ALERT below
  the AUC the champion was accepted with.
A retrain is only a candidate: promotion still goes through the governance gate.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.config.settings import settings


def decide_retraining(
    drift_report: Optional[Dict[str, Any]] = None,
    feedback_reports: Iterable[Optional[Dict[str, Any]]] = (),
) -> Dict[str, Any]:
    reasons: List[str] = []
    targets: List[str] = []

    if drift_report and drift_report.get("requires_retraining"):
        target = drift_report.get("target", "purchase")
        reasons.append(
            f"{target}: feature/prediction drift {drift_report.get('overall_status')} "
            f"(drift ratio {drift_report.get('feature_drift_summary', {}).get('drift_ratio')})"
        )
        targets.append(target)

    for report in feedback_reports:
        if report and report.get("degraded"):
            reasons.append(
                f"{report['target']}: realized AUC {report['realized_roc_auc']:.4f} is "
                f"{report['auc_drop']:.4f} below the accepted {report['expected_roc_auc']:.4f}"
            )
            targets.append(report["target"])

    retrain = bool(reasons) and settings.AUTO_RETRAIN_ON_DEGRADATION
    return {
        "retrain": retrain,
        "targets": sorted(set(targets)),
        "reasons": reasons,
        "auto_retrain_enabled": settings.AUTO_RETRAIN_ON_DEGRADATION,
    }
