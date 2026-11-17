"""
Model Registry and Champion / Challenger Governance Module.
Manages model promotion gates, artifact bundling, and versioned persistence.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import lightgbm as lgb

from src.config.settings import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Automates Champion / Challenger evaluation, quality gates, and model deployment packaging.
    """

    def __init__(self, registry_dir: Optional[Path | str] = None) -> None:
        self.registry_dir = Path(registry_dir or (settings.BASE_DIR / "models"))
        self.history_dir = self.registry_dir / "history"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_governance_gate(
        self,
        candidate_metrics: Dict[str, float],
        min_auc: float = 0.75,
        max_brier: float = 0.20,
        min_lift_10: float = 1.50,
    ) -> Dict[str, Any]:
        """
        Enforces strict production quality gates before model promotion.
        """
        auc_pass = candidate_metrics["roc_auc"] >= min_auc
        brier_pass = candidate_metrics["brier_score"] <= max_brier
        lift_pass = candidate_metrics["lift_at_10pct"] >= min_lift_10

        passed_all = auc_pass and brier_pass and lift_pass
        gate_result = {
            "passed": passed_all,
            "checks": {
                "roc_auc": {"value": candidate_metrics["roc_auc"], "threshold": min_auc, "pass": auc_pass},
                "brier_score": {"value": candidate_metrics["brier_score"], "threshold": max_brier, "pass": brier_pass},
                "lift_at_10pct": {"value": candidate_metrics["lift_at_10pct"], "threshold": min_lift_10, "pass": lift_pass},
            },
        }

        if passed_all:
            logger.info("Candidate Model PASSED all governance quality gates.")
        else:
            logger.warning(f"Candidate Model FAILED governance gates: {gate_result['checks']}")

        return gate_result

    def register_and_promote(
        self,
        model: Any,
        features: List[str],
        metrics: Dict[str, float],
        calibrator: Optional[Any] = None,
        fit_cutoffs: Optional[List[int]] = None,
        val_cutoffs: Optional[List[int]] = None,
        bundle_name: str = "18 + chu kỳ",
    ) -> Path:
        """
        Packages model bundle into champion_model.joblib and creates timestamped archive.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        package = {
            "bundles": {
                bundle_name: {
                    "model": model,
                    "calibrator": calibrator,
                    "features": features,
                }
            },
            "epsilon": 1e-4,
            "lightgbm_version": lgb.__version__,
            "fit_cutoffs": fit_cutoffs or [],
            "calibration_cutoffs": val_cutoffs or [],
            "metrics": metrics,
            "promoted_at": timestamp,
        }

        # 1. Save timestamped archive
        archive_path = self.history_dir / f"champion_model_{timestamp}.joblib"
        joblib.dump(package, archive_path)
        logger.info(f"Archived model bundle to {archive_path}")

        # 2. Promote as active Champion Model
        target_path = self.registry_dir / "champion_model.joblib"
        joblib.dump(package, target_path)
        logger.info(f"Promoted active Champion Model to {target_path}")

        # 3. Write metadata manifest
        meta_path = self.registry_dir / "latest_model_manifest.json"
        manifest = {
            "promoted_at": timestamp,
            "bundle_name": bundle_name,
            "features": features,
            "metrics": metrics,
            "fit_cutoffs": fit_cutoffs,
            "val_cutoffs": val_cutoffs,
            "archive_path": str(archive_path),
        }
        meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return target_path
