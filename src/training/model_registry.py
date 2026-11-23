"""
Model Registry: bundle packaging, promotion gate and versioned persistence.

Bundle schema v2 is self-describing: it carries the feature contract version, feature
and categorical lists, prediction horizon, temporal splits, the data fingerprint and the
git commit that produced it, so any champion can be traced and re-trained from this repo.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
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
    created_at = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_name": bundle_name or f"lgbm_{config.target}_{config.feature_set_version}_h{config.horizon_days}_{created_at}",
        "created_at": created_at,
        "target": config.target,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write to a temp file in the same directory, then rename: readers never see a partial file."""
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _manifest_for(package: Dict[str, Any]) -> Dict[str, Any]:
    if package.get("schema_version", 1) >= 2:
        return bundle_manifest(package)
    bundles = package.get("bundles") or {}
    name, inner = next(iter(bundles.items())) if bundles else ("unknown", {})
    return {
        "schema_version": 1,
        "bundle_name": name,
        "features": inner.get("features", []),
        "horizon_days": package.get("prediction_window_days"),
        "metrics": package.get("metrics", {}),
        "fit_cutoffs": package.get("fit_cutoffs"),
        "val_cutoffs": package.get("val_cutoffs"),
        "test_cutoffs": package.get("test_cutoffs"),
    }


def registry_dir_for(config: TrainingConfig) -> Path:
    """models/ for the purchase champion (historical location), models/<subdir>/ otherwise."""
    base = settings.BASE_DIR / "models"
    return base / config.registry_subdir if config.registry_subdir else base


class ModelRegistry:
    """
    File-based registry:
      champion_model.joblib                     active champion (replaced atomically)
      latest_model_manifest.json                JSON view of the active champion
      history/champion_model_<version>.joblib   every bundle that has been champion
      history/promotions.jsonl                  append-only audit log (promotions, rollbacks)
    A version id is the timestamp suffix of its history file; ids sort chronologically.
    """

    ARCHIVE_PREFIX = "champion_model_"

    def __init__(self, registry_dir: Optional[Path | str] = None) -> None:
        self.registry_dir = Path(registry_dir or (settings.BASE_DIR / "models"))
        self.history_dir = self.registry_dir / "history"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    @property
    def champion_path(self) -> Path:
        return self.registry_dir / "champion_model.joblib"

    @property
    def manifest_path(self) -> Path:
        return self.registry_dir / "latest_model_manifest.json"

    @property
    def audit_log_path(self) -> Path:
        return self.history_dir / "promotions.jsonl"

    # ------------------------------------------------------------------ gate
    def evaluate_governance_gate(
        self, candidate_metrics: Dict[str, float], gate: Optional[GateConfig] = None
    ) -> Dict[str, Any]:
        """Absolute, base-rate-relative quality checks on out-of-time test metrics."""
        g = gate or GateConfig()
        m = candidate_metrics
        rules = [  # (metric, threshold, higher_is_better); None threshold = check disabled
            ("roc_auc", g.min_auc, True),
            ("brier_skill_score", g.min_brier_skill, True),
            ("normalized_lift_10", g.min_normalized_lift_10, True),
            ("lift_at_10pct", g.min_lift_10, True),
            ("pr_auc_gain", g.min_pr_auc_gain, True),
            ("ece", g.max_ece, False),
        ]
        checks = {
            metric: {
                "value": m[metric],
                "threshold": threshold,
                "pass": m[metric] >= threshold if higher else m[metric] <= threshold,
            }
            for metric, threshold, higher in rules
            if threshold is not None
        }
        passed = all(c["pass"] for c in checks.values())
        if passed:
            logger.info("Candidate PASSED the quality gate.")
        else:
            failed = {k: v for k, v in checks.items() if not v["pass"]}
            logger.warning(f"Candidate FAILED the quality gate: {failed}")
        return {"passed": passed, "checks": checks}

    # -------------------------------------------------------------- versions
    def save_candidate(self, bundle: Dict[str, Any], directory: Path | str) -> Path:
        """Persists a bundle outside the registry (no promotion)."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"candidate_{bundle['created_at']}.joblib"
        joblib.dump(bundle, path)
        return path

    def _archive_path(self, version: str) -> Path:
        return self.history_dir / f"{self.ARCHIVE_PREFIX}{version}.joblib"

    def list_versions(self) -> List[str]:
        return sorted(
            p.stem[len(self.ARCHIVE_PREFIX):] for p in self.history_dir.glob(f"{self.ARCHIVE_PREFIX}*.joblib")
        )

    def current_version(self) -> Optional[str]:
        """History version whose bytes equal the active champion, if it is archived."""
        if not self.champion_path.exists():
            return None
        current = _sha256(self.champion_path)
        for version in reversed(self.list_versions()):
            if _sha256(self._archive_path(version)) == current:
                return version
        return None

    def _archive_outgoing_champion(self) -> None:
        """Keeps the champion being replaced roll-back-able even if it predates the registry."""
        if not self.champion_path.exists() or self.current_version() is not None:
            return
        mtime = datetime.datetime.fromtimestamp(self.champion_path.stat().st_mtime)
        version = f"{mtime.strftime('%Y%m%d_%H%M%S_%f')}_pre_registry"
        shutil.copy2(self.champion_path, self._archive_path(version))
        logger.info(f"Archived unregistered outgoing champion as version {version}")

    def _activate(self, version: str, event: str, reason: str) -> Path:
        archive = self._archive_path(version)
        _atomic_write_bytes(self.champion_path, archive.read_bytes())
        manifest = _manifest_for(joblib.load(archive))
        manifest.update({
            "version": version,
            "activated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "activation_event": event,
            "activation_reason": reason,
            "archive_path": str(archive),
        })
        _atomic_write_bytes(self.manifest_path, json.dumps(manifest, indent=2, default=str).encode("utf-8"))
        entry = {"event": event, "version": version, "bundle_name": manifest.get("bundle_name"),
                 "at": manifest["activated_at"], "reason": reason}
        with self.audit_log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
        return self.champion_path

    def register_and_promote(self, bundle: Dict[str, Any], reason: str = "governance gate passed") -> Path:
        """Archives the bundle and atomically makes it the champion."""
        if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Refusing to promote bundle with schema_version={bundle.get('schema_version')}")
        self._archive_outgoing_champion()
        version = bundle["created_at"]
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer)
        _atomic_write_bytes(self._archive_path(version), buffer.getvalue())
        path = self._activate(version, "promote", reason)
        logger.info(f"Promoted '{bundle['bundle_name']}' as version {version}")
        return path

    def rollback(self, to_version: Optional[str] = None, reason: str = "manual rollback") -> str:
        """Re-activates `to_version`, or the version archived before the current champion."""
        versions = self.list_versions()
        if to_version is None:
            current = self.current_version()
            if current is None:
                raise ValueError("Active champion is not in the registry history; pass to_version explicitly")
            older = [v for v in versions if v < current]
            if not older:
                raise ValueError(f"No version older than the current champion {current} to roll back to")
            to_version = older[-1]
        elif to_version not in versions:
            raise ValueError(f"Unknown version {to_version}; available: {versions}")
        self._activate(to_version, "rollback", reason)
        logger.warning(f"Rolled champion back to version {to_version} ({reason})")
        return to_version
