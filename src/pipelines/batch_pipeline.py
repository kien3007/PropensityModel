"""
Batch Pipeline Orchestration Module.
Executes Master Dataset Ingestion, Lakehouse Transformations, Point-in-Time Feature Extraction,
Champion Model Scoring, and Serving Layer Cache Population.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.config.settings import settings
from src.features.pit_extractor import PointInTimeFeatureExtractor
from src.lakehouse.engine import LakehouseEngine
from src.models.predictor import PropensityPredictor
from src.monitoring.drift_detector import DataDriftDetector
from src.monitoring.metrics import record_drift_metrics
from src.speed_layer.delta_stream import SpeedLayerProcessor

logger = logging.getLogger("BatchPipeline")


def run_batch_pipeline(
    cutoff_day: int = 711,
    transactions_csv: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    lakehouse_path: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Executes the End-to-End Batch Layer Pipeline.
    """
    start_time = datetime.datetime.now()
    run_id = f"BATCH_{cutoff_day}_{start_time.strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"=== Starting Enterprise Batch Scoring Pipeline: {run_id} (Cutoff DAY={cutoff_day}) ===")

    out_dir = Path(output_dir or settings.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tx_csv = Path(transactions_csv or (settings.DATA_RAW_DIR / "transaction_data.csv"))
    if not tx_csv.exists():
        raise FileNotFoundError(f"Source transactions file not found at {tx_csv}")

    # 1. Initialize Lakehouse & Bronze Ingestion
    lakehouse = LakehouseEngine(lakehouse_path=lakehouse_path)
    logger.info("Verifying Lakehouse Bronze/Silver/Gold layers...")
    lakehouse.ingest_bronze()
    lakehouse.build_silver_layer()
    lakehouse.build_gold_layer()

    # 2. Extract Point-in-Time Features
    logger.info(f"Extracting Point-in-Time features up to DAY={cutoff_day}...")
    predictor = PropensityPredictor(model_path=model_path)

    if len(predictor.features) == 28:
        snapshot = PointInTimeFeatureExtractor.extract_multidomain_features(
            cutoff=cutoff_day,
            raw_data_dir=tx_csv.parent,
        )
    else:
        cols = ["household_key", "BASKET_ID", "DAY", "SALES_VALUE"]
        tx = pd.read_csv(tx_csv, usecols=cols)
        snapshot = PointInTimeFeatureExtractor.extract_features(
            transactions=tx,
            cutoff=cutoff_day,
            feature_cols=predictor.features,
        )

    # 3. Model Inference & Business Tiering
    logger.info(f"Scoring 2,500 households with Champion model '{predictor.bundle_name}'...")
    scores_df = predictor.score_households(snapshot)

    # 4. Save to Lakehouse Gold Layer
    records_saved = lakehouse.save_batch_scores(scores_df, run_id=run_id, snapshot_day=cutoff_day)
    logger.info(f"Stored {records_saved} batch scores in Gold Lakehouse table.")

    # 5. Export Multi-Channel Artifacts
    parquet_path = out_dir / f"household_scores_day_{cutoff_day}.parquet"
    csv_path = out_dir / f"household_scores_day_{cutoff_day}.csv"
    crm_csv_path = out_dir / f"crm_export_day_{cutoff_day}.csv"

    scores_df.to_parquet(parquet_path, index=False)
    scores_df.to_csv(csv_path, index=False)

    crm_cols = ["household_key", "purchase_probability", "rank", "decile", "urgency_tier", "marketing_action"]
    scores_df[crm_cols].to_csv(crm_csv_path, index=False)

    # 6. Evaluate Model & Data Drift against Gold Baseline
    drift_summary = None
    gold_snapshot_file = settings.BASE_DIR / "data" / "lakehouse" / "gold" / "fct_household_features_snapshot.parquet"
    if gold_snapshot_file.exists():
        try:
            ref_snapshot = pd.read_parquet(gold_snapshot_file)
            detector = DataDriftDetector(output_dir=out_dir / "reports")
            drift_report = detector.generate_full_drift_audit(
                reference_df=ref_snapshot,
                current_df=snapshot,
                current_scores=scores_df["purchase_probability"].to_numpy(),
                run_id=f"DRIFT_{run_id}",
            )
            psi_dict = {f: details["psi"] for f, details in drift_report["feature_details"].items()}
            record_drift_metrics(psi_dict)
            drift_summary = {
                "status": drift_report["overall_status"],
                "drift_ratio": drift_report["feature_drift_summary"]["drift_ratio"],
                "requires_retraining": drift_report["requires_retraining"],
            }
            logger.info(f"Drift Audit: {drift_summary['status']} (Requires Retrain={drift_summary['requires_retraining']})")
        except Exception as e:
            logger.warning(f"Drift Audit could not complete: {e}")

    # 7. Reset Speed Layer Intraday Deltas for fresh cycle
    speed_proc = SpeedLayerProcessor()
    speed_proc.reset_intraday_deltas()

    duration = (datetime.datetime.now() - start_time).total_seconds()

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "cutoff_day": cutoff_day,
        "model_bundle": predictor.bundle_name,
        "feature_count": len(predictor.features),
        "total_households_scored": len(scores_df),
        "mean_propensity": float(scores_df["purchase_probability"].mean()),
        "top_decile_threshold": float(scores_df[scores_df["decile"] == 10]["purchase_probability"].min()),
        "top_decile_count": int((scores_df["decile"] == 10).sum()),
        "urgent_count": int((scores_df["urgency_tier"] == "URGENT").sum()),
        "lakehouse_records_saved": records_saved,
        "drift_audit": drift_summary,
        "duration_seconds": round(duration, 2),
        "artifacts": {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "crm_csv": str(crm_csv_path),
        },
    }

    manifest_path = out_dir / f"batch_manifest_day_{cutoff_day}.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"=== Enterprise Batch Pipeline Completed in {duration:.2f}s ===")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Enterprise Batch Scoring Pipeline")
    parser.add_argument("--cutoff", type=int, default=711, help="Cutoff snapshot day (default: 711)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_batch_pipeline(cutoff_day=args.cutoff)
