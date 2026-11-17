"""
End-to-End Retraining Pipeline Orchestration Module.
Extracts Point-in-Time temporal datasets, trains Champion Candidate (LightGBM),
calibrates probabilities, evaluates governance quality gates, and registers promoted model.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config.settings import settings
from src.features.pit_extractor import MODEL_FEATURES
from src.training.dataset_builder import DatasetBuilder
from src.training.trainer import ModelTrainer
from src.training.calibrator import PlattCalibrator
from src.training.evaluator import ModelEvaluator
from src.training.model_registry import ModelRegistry

logger = logging.getLogger("TrainPipeline")


def run_training_pipeline(
    train_cutoffs: Optional[List[int]] = None,
    val_cutoff: int = 537,
    horizon_days: int = 7,
    transactions_csv: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    force_promote: bool = False,
) -> Dict[str, Any]:
    """
    Executes the End-to-End Retraining Pipeline.
    """
    start_time = datetime.datetime.now()
    train_cutoffs = train_cutoffs or [250, 300, 350, 400, 450]
    run_id = f"TRAIN_{start_time.strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"=== Starting Model Retraining Pipeline: {run_id} ===")
    logger.info(f"Train Cutoffs: {train_cutoffs} | Validation Cutoff: {val_cutoff} | Horizon: {horizon_days}d")

    out_dir = Path(output_dir or settings.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tx_path = Path(transactions_csv or (settings.DATA_RAW_DIR / "transaction_data.csv"))
    if not tx_path.exists():
        raise FileNotFoundError(f"Source transactions file not found at {tx_path}")

    # 1. Load transactions
    cols = ["household_key", "BASKET_ID", "DAY", "SALES_VALUE"]
    logger.info(f"Loading raw transactions from {tx_path}...")
    tx = pd.read_csv(tx_path, usecols=cols)

    # 2. Build temporal training dataset
    logger.info("Building multi-cutoff training dataset...")
    train_df = DatasetBuilder.build_multi_cutoff_dataset(
        transactions=tx,
        cutoffs=train_cutoffs,
        horizon_days=horizon_days,
        feature_cols=MODEL_FEATURES,
    )

    # 3. Build validation dataset
    logger.info(f"Building validation dataset at DAY={val_cutoff}...")
    val_df = DatasetBuilder.build_snapshot_with_target(
        transactions=tx,
        cutoff_day=val_cutoff,
        horizon_days=horizon_days,
        feature_cols=MODEL_FEATURES,
    )

    X_train, y_train = train_df[MODEL_FEATURES], train_df["target"]
    X_val, y_val = val_df[MODEL_FEATURES], val_df["target"]

    # 4. Train Model
    trainer = ModelTrainer(features=MODEL_FEATURES)
    lgbm_model = trainer.train_lightgbm(X_train, y_train)

    # 5. Evaluate Raw Predictions
    raw_val_probs = lgbm_model.predict_proba(X_val)[:, 1]
    raw_metrics = ModelEvaluator.evaluate(y_val, raw_val_probs)

    # 6. Fit Platt Calibrator
    calibrator = PlattCalibrator()
    calibrator.fit(raw_val_probs, y_val.to_numpy())
    calibrated_val_probs = calibrator.predict_proba(raw_val_probs)
    calibrated_metrics = ModelEvaluator.evaluate(y_val, calibrated_val_probs)

    # 7. Model Registry Governance Gate & Promotion
    registry = ModelRegistry()
    gate_result = registry.evaluate_governance_gate(calibrated_metrics)

    should_promote = gate_result["passed"] or force_promote
    promoted_path = None
    if should_promote:
        promoted_path = registry.register_and_promote(
            model=lgbm_model,
            features=MODEL_FEATURES,
            metrics=calibrated_metrics,
            calibrator=calibrator.calibrator,
            fit_cutoffs=train_cutoffs,
            val_cutoffs=[val_cutoff],
        )

    # 8. Export Feature Importances
    fi_df = trainer.get_feature_importances(lgbm_model)
    fi_csv_path = out_dir / "latest_feature_importances.csv"
    fi_df.to_csv(fi_csv_path, index=False)

    duration = (datetime.datetime.now() - start_time).total_seconds()

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_cutoffs": train_cutoffs,
        "val_cutoff": val_cutoff,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "governance_gate": gate_result,
        "promoted": should_promote,
        "promoted_path": str(promoted_path) if promoted_path else None,
        "feature_importances_path": str(fi_csv_path),
        "duration_seconds": round(duration, 2),
    }

    manifest_path = out_dir / f"training_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"=== Model Retraining Pipeline Finished in {duration:.2f}s (Promoted: {should_promote}) ===")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Automated Propensity Retraining Pipeline")
    parser.add_argument(
        "--train-cutoffs",
        type=str,
        default="250,300,350,400,450",
        help="Comma-separated training cutoff days (default: 250,300,350,400,450)",
    )
    parser.add_argument("--val-cutoff", type=int, default=537, help="Validation cutoff day (default: 537)")
    parser.add_argument("--horizon", type=int, default=7, help="Prediction horizon in days (default: 7)")
    parser.add_argument("--force-promote", action="store_true", help="Force promotion even if gate fails")
    args = parser.parse_args()

    cutoffs = [int(c.strip()) for c in args.train_cutoffs.split(",")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_training_pipeline(
        train_cutoffs=cutoffs,
        val_cutoff=args.val_cutoff,
        horizon_days=args.horizon,
        force_promote=args.force_promote,
    )
