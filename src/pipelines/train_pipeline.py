"""
End-to-End Retraining Pipeline.

  load data (DataLoader) -> fit / val / test splits (DatasetBuilder, TrainingConfig cutoffs)
  -> LightGBM with early stopping on val -> calibrator chosen on val (cross-fitted ECE)
  -> out-of-time evaluation on test -> governance gate -> bundle (+ optional promotion)

Governance gate = quality checks (base-rate relative, src/config/model_config.GateConfig)
AND non-inferiority versus the active champion on the same test rows.

Every run writes a candidate bundle and a JSON manifest to `output_dir`, whether or not
it is promoted, so any run can be inspected or compared against the champion later.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.model_config import DEFAULT_TRAINING_CONFIG, TRAINING_CONFIGS, TrainingConfig
from src.config.settings import settings
from src.data.loader import DataLoader, RawDataBundle
from src.features.contract import CATEGORICAL_FEATURES, features_for
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.features.targets import get_target
from src.training.calibrator import select_calibrator
from src.training.champion_challenger import compare_with_champion
from src.training.dataset_builder import DatasetBuilder
from src.training.evaluator import ModelEvaluator
from src.training.model_registry import ModelRegistry, build_bundle, registry_dir_for
from src.training.trainer import ModelTrainer

logger = logging.getLogger("TrainPipeline")


def run_training_pipeline(
    config: Optional[TrainingConfig] = None,
    data: Optional[RawDataBundle] = None,
    output_dir: Optional[Path | str] = None,
    registry_dir: Optional[Path | str] = None,
    promote: bool = True,
    force_promote: bool = False,
    feature_source: Optional[str] = None,
    feature_store_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Trains and evaluates a champion candidate.

    promote=False never touches the registry (used by the reproduction check);
    force_promote=True promotes even when the governance gate fails (manual use only).
    """
    cfg = config or DEFAULT_TRAINING_CONFIG
    model_features = features_for(cfg.feature_set_version)  # any supported feature set can be trained

    start = datetime.datetime.now()
    run_id = f"TRAIN_{start.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(output_dir or settings.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== Retraining {run_id}: target={cfg.target}, horizon={cfg.horizon_days}d, "
                f"fit={list(cfg.fit_cutoffs)}, "
                f"val={list(cfg.val_cutoffs)}, test={list(cfg.test_cutoffs)} ===")

    # 1. Data & temporal splits
    raw = data or DataLoader.load()
    builder = MultiDomainFeatureBuilder(raw, lookback_days=cfg.lookback_days,
                                        feature_set_version=cfg.feature_set_version)
    source_name = (feature_source or settings.TRAINING_FEATURE_SOURCE).lower()
    if source_name == "feast":
        # Point-in-time reads from the historical feature store; cutoffs never snapshotted
        # are computed once and written back (backfill), labels still come from baskets.
        from src.feature_store.client import FeatureStoreClient, FeatureStoreSource

        features = FeatureStoreSource(builder, FeatureStoreClient(root=feature_store_dir,
                                                                  feature_set_version=cfg.feature_set_version))
    elif source_name == "builder":
        features = builder
    else:
        raise ValueError(f"Unknown feature_source '{source_name}' (expected 'feast' or 'builder')")
    splits = DatasetBuilder(features).build_splits(cfg)
    fit, val, test = splits["fit"], splits["val"], splits["test"]

    # 2. Train with early stopping on validation
    trainer = ModelTrainer(
        features=model_features,
        categorical_features=CATEGORICAL_FEATURES,
        lgbm_params=cfg.lgbm_params,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    model = trainer.train_lightgbm(fit.X, fit.y, val.X, val.y)
    best_iteration = trainer.best_iteration(model)

    # 3. Calibration chosen on validation only
    raw_val = trainer.predict_raw(model, val.X)
    calibrator, calibration_report = select_calibrator(
        raw_val, val.y.to_numpy(), methods=cfg.calibration_methods, n_folds=cfg.calibration_cv_folds,
    )
    # Raw (uncalibrated) metrics: the calibrator is fitted on these same rows, so calibrated
    # validation metrics would be in-sample. Its out-of-fold ECE is in calibration_report.
    val_metrics = ModelEvaluator.evaluate(val.y, raw_val)

    # 4. Out-of-time test evaluation (LightGBM + calibrator)
    raw_test = trainer.predict_raw(model, test.X)
    test_metrics = ModelEvaluator.evaluate(test.y, calibrator.predict_proba(raw_test))
    test_metrics_uncalibrated = ModelEvaluator.evaluate(test.y, raw_test)

    # 5. Bundle, gate, promotion
    bundle = build_bundle(
        model=model,
        calibrator=calibrator,
        features=model_features,
        categorical_features=CATEGORICAL_FEATURES,
        config=cfg,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        data_fingerprint=raw.fingerprint(),
        best_iteration=best_iteration,
        calibration_report=calibration_report,
    )
    registry = ModelRegistry(registry_dir=registry_dir or registry_dir_for(cfg))
    candidate_path = registry.save_candidate(bundle, out_dir / "candidates")
    quality = registry.evaluate_governance_gate(test_metrics, cfg.gate)
    comparison = compare_with_champion(
        challenger_probs=calibrator.predict_proba(raw_test),
        test_frame=test.frame,
        champion_path=registry.champion_path,
        config=cfg,
        # A champion on another feature set is scored with its own features on the same cutoffs.
        champion_test_frame=lambda version: DatasetBuilder(builder.for_version(version)).build(
            cfg.test_cutoffs, cfg.horizon_days, get_target(cfg.target)),
    )
    gate = {"passed": quality["passed"] and comparison["passed"], "quality": quality, "champion_comparison": comparison}

    should_promote = promote and (gate["passed"] or force_promote)
    promoted_path = None
    if should_promote:
        reason = "governance gate passed" if gate["passed"] else "force_promote override (gate failed)"
        promoted_path = registry.register_and_promote(bundle, reason=reason)
    elif promote:
        blockers = [k for k in ("quality", "champion_comparison") if not gate[k]["passed"]]
        logger.warning(f"Candidate NOT promoted: failed {blockers} (force_promote overrides).")

    # Out-of-time predictions for downstream business evaluation (scripts/target_analysis.py).
    predictions = test.frame[["household_key", "cutoff_day", "target"]].copy()
    predictions["model_probability"] = calibrator.predict_proba(raw_test)
    predictions_path = out_dir / f"test_predictions_{cfg.target}_{run_id}.parquet"
    predictions.to_parquet(predictions_path, index=False)

    fi_path = out_dir / f"latest_feature_importances_{cfg.target}.csv"
    trainer.get_feature_importances(model).to_csv(fi_path, index=False)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "target": cfg.target,
        "bundle_name": bundle["bundle_name"],
        "config": cfg.to_dict(),
        "data_source": raw.source,
        "feature_source": source_name,
        "data_fingerprint": bundle["data_fingerprint"],
        "git_sha": bundle["git_sha"],
        "rows": {name: len(s.frame) for name, s in splits.items()},
        "positive_rate": {name: round(s.positive_rate, 4) for name, s in splits.items()},
        "best_iteration": best_iteration,
        "calibration_method": calibrator.method,
        "calibration_report": calibration_report,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_uncalibrated": test_metrics_uncalibrated,
        "governance_gate": gate,
        "promoted": promoted_path is not None,
        "promoted_path": str(promoted_path) if promoted_path else None,
        "candidate_path": str(candidate_path),
        "feature_importances_path": str(fi_path),
        "test_predictions_path": str(predictions_path),
        "duration_seconds": round((datetime.datetime.now() - start).total_seconds(), 2),
    }
    (out_dir / f"training_manifest_{run_id}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"=== {run_id} finished in {summary['duration_seconds']}s "
                f"(test AUC={test_metrics['roc_auc']:.4f}, promoted={summary['promoted']}) ===")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the propensity retraining pipeline")
    parser.add_argument("--target", choices=sorted(TRAINING_CONFIGS), default="purchase")
    parser.add_argument("--no-promote", action="store_true", help="Train and evaluate only; never touch the registry")
    parser.add_argument("--force-promote", action="store_true", help="Promote even if the governance gate fails")
    parser.add_argument("--data-source", choices=["lakehouse", "csv"], default=None)
    parser.add_argument("--feature-source", choices=["feast", "builder"], default=None,
                        help="Where training features come from (default: settings.TRAINING_FEATURE_SOURCE)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_training_pipeline(
        config=TRAINING_CONFIGS[args.target],
        data=DataLoader.load(args.data_source),
        feature_source=args.feature_source,
        promote=not args.no_promote,
        force_promote=args.force_promote,
    )
