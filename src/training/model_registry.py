"""
Model Registry: bundle packaging, promotion gate and versioned persistence.

Bundle schema v2 is self-describing: it carries the feature contract version, feature
and categorical lists, prediction horizon, temporal splits, the data fingerprint and the
git commit that produced it, so any champion can be traced and re-trained from this repo.
"""
from __future__ import annotations

import datetime
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import sklearn

from src.config.model_config import GateConfig, TrainingConfig
from src.config.settings import settings
from src.training.calibrator import BaseCalibrator

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = 2


def current_git_sha(repo_dir: Optional[Path] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir or settings.BASE_DIR, capture_output=True, text=True, timeout=5, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir or settings.BASE_DIR, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_bundle(
    model: lgb.LGBMClassifier,
    calibrator: BaseCalibrator,
    features: List[str],
    categorical_features: List[str],
    config: TrainingConfig,
    test_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    data_fingerprint: str,
    best_iteration: int,
    calibration_report: Optional[Dict[str, Any]] = None,
    bundle_name: Optional[str] = None,
) -> Dict[str, Any]:
    created_at = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_name": bundle_name or f"lgbm_{config.feature_set_version}_h{config.horizon_days}_{created_at}",
        "created_at": created_at,
        "feature_set_version": config.feature_set_version,
        "features": list(features),
        "categorical_features": list(categorical_features),
        "horizon_days": config.horizon_days,
        "lookback_days": config.lookback_days,
        "fit_cutoffs": list(config.fit_cutoffs),
        "val_cutoffs": list(config.val_cutoffs),
        "test_cutoffs": list(config.test_cutoffs),
        "model": model,
        "calibrator": calibrator,
        "calibration_method": calibrator.method,
        "calibration_report": calibration_report or {},
        "best_iteration": int(best_iteration),
        "lgbm_params": dict(config.lgbm_params),
        "metrics": test_metrics,
        "val_metrics": val_metrics,
        "data_fingerprint": data_fingerprint,
        "git_sha": current_git_sha(),
        "lightgbm_version": lgb.__version__,
        "sklearn_version": sklearn.__version__,
    }


def bundle_manifest(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-serialisable view of a bundle (everything except fitted objects)."""
    return {k: v for k, v in bundle.items() if k not in ("model", "calibrator")}


class ModelRegistry:
    def __init__(self, registry_dir: Optional[Path | str] = None) -> None:
        self.registry_dir = Path(registry_dir or (settings.BASE_DIR / "models"))
        self.history_dir = self.registry_dir / "history"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    @property
    def champion_path(self) -> Path:
        return self.registry_dir / "champion_model.joblib"

    def evaluate_governance_gate(self, candidate_metrics: Dict[str, float], gate: Optional[GateConfig] = None) -> Dict[str, Any]:
        """Absolute quality thresholds on out-of-time test metrics."""
        g = gate or GateConfig()
        checks = {
            "roc_auc": {"value": candidate_metrics["roc_auc"], "threshold": g.min_auc,
                        "pass": candidate_metrics["roc_auc"] >= g.min_auc},
            "brier_score": {"value": candidate_metrics["brier_score"], "threshold": g.max_brier,
                            "pass": candidate_metrics["brier_score"] <= g.max_brier},
            "lift_at_10pct": {"value": candidate_metrics["lift_at_10pct"], "threshold": g.min_lift_10,
                              "pass": candidate_metrics["lift_at_10pct"] >= g.min_lift_10},
        }
        passed = all(c["pass"] for c in checks.values())
        if passed:
            logger.info("Candidate model PASSED all governance gates.")
        else:
            logger.warning(f"Candidate model FAILED governance gates: {checks}")
        return {"passed": passed, "checks": checks}

    def save_candidate(self, bundle: Dict[str, Any], directory: Path | str) -> Path:
        """Persists a bundle outside the registry (no promotion)."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"candidate_{bundle['created_at']}.joblib"
        joblib.dump(bundle, path)
        return path

    def register_and_promote(self, bundle: Dict[str, Any]) -> Path:
        """Archives the bundle, promotes it to champion and writes the manifest."""
        if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Refusing to promote bundle with schema_version={bundle.get('schema_version')}")
        stamp = bundle["created_at"]

        archive_path = self.history_dir / f"champion_model_{stamp}.joblib"
        joblib.dump(bundle, archive_path)
        joblib.dump(bundle, self.champion_path)

        manifest = bundle_manifest(bundle)
        manifest["promoted_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest["archive_path"] = str(archive_path)
        (self.registry_dir / "latest_model_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        logger.info(f"Promoted '{bundle['bundle_name']}' to {self.champion_path} (archive: {archive_path})")
        return self.champion_path
