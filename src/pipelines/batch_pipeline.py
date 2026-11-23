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
from src.config.model_config import LAPSE_TRAINING_CONFIG
from src.data.loader import DataLoader, RawDataBundle
from src.features.contract import FEATURE_SET_VERSION
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.features.targets import LAPSE
from src.lakehouse.engine import LakehouseEngine
from src.models.predictor import PropensityPredictor
from src.monitoring.drift_detector import DataDriftDetector
from src.monitoring.reference import build_champion_reference
from src.monitoring.state import MonitoringState
from src.speed_layer.delta_stream import SpeedLayerProcessor
from src.serving.batch_store import BatchScoreStore
from src.training.model_registry import registry_dir_for

logger = logging.getLogger("BatchPipeline")


def run_batch_pipeline(
    cutoff_day: int = 711,
    raw_data_dir: Optional[Path | str] = None,
    data: Optional[RawDataBundle] = None,
    output_dir: Optional[Path | str] = None,
    lakehouse_path: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
    lapse_model_path: Optional[Path | str] = None,
    challenger_model_path: Optional[Path | str] = None,
    batch_store: Optional[BatchScoreStore] = None,
    monitoring_state: Optional[MonitoringState] = None,
    feature_store_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Executes the End-to-End Batch Layer Pipeline.
    """
    start_time = datetime.datetime.now()
    run_id = f"BATCH_{cutoff_day}_{start_time.strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"=== Starting Enterprise Batch Scoring Pipeline: {run_id} (Cutoff DAY={cutoff_day}) ===")

    out_dir = Path(output_dir or settings.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Lakehouse & Bronze Ingestion
    lakehouse = LakehouseEngine(lakehouse_path=lakehouse_path, raw_data_dir=raw_data_dir)
    logger.info("Verifying Lakehouse Bronze/Silver/Gold layers...")
    lakehouse.ingest_bronze()
    lakehouse.build_silver_layer()
    lakehouse.build_gold_layer()

    # 2. Extract Point-in-Time Features through the same loader + builder used in training
    predictor = PropensityPredictor(model_path=model_path, expected_target="purchase")
    if predictor.feature_set_version != FEATURE_SET_VERSION:
        raise ValueError(
            f"Champion uses feature set {predictor.feature_set_version}; batch scoring implements {FEATURE_SET_VERSION}"
        )
    data = data or DataLoader.from_lakehouse(lakehouse, ingest=False)
    logger.info(f"Extracting Point-in-Time features up to DAY={cutoff_day} from {data.source}...")
    builder = MultiDomainFeatureBuilder(data)
    snapshot = builder.features_at(cutoff_day)

    # 2b. Persist the snapshot in the historical feature store (offline history + online refresh)
    feature_store_summary = None
    if settings.FEATURE_STORE_ENABLED:
        try:
            from src.feature_store.client import FeatureStoreClient

            fs = FeatureStoreClient(root=feature_store_dir)
            rows = fs.write_snapshot(snapshot, cutoff_day, online=True)
            feature_store_summary = {"rows": rows, "snapshot_days": len(fs.snapshot_days()), "root": str(fs.root)}
        except Exception as e:
            if settings.STRICT_INFRA:
                raise
            logger.error(f"Feature store write failed for DAY {cutoff_day}: {e}")

    # 3. Model Inference & Business Tiering
    logger.info(f"Scoring {len(snapshot):,} households with Champion model '{predictor.bundle_name}'...")
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

    # 5b. Publish batch views to the serving layer (Redis when available) for every replica
    store = batch_store or BatchScoreStore()
    store.publish(scores_df, "champion", {"run_id": run_id, "snapshot_day": cutoff_day,
                                          "model_bundle": predictor.bundle_name})

    # 5c. Canary challenger: score the same snapshot so routed traffic gets its own batch view
    challenger_summary = None
    challenger_path = challenger_model_path or settings.CHALLENGER_MODEL_PATH
    if challenger_path:
        challenger = PropensityPredictor(model_path=challenger_path, expected_target="purchase",
                                         expected_horizon_days=predictor.horizon_days)
        challenger_scores = challenger.score_households(snapshot)
        challenger_parquet = out_dir / f"household_scores_challenger_day_{cutoff_day}.parquet"
        challenger_scores.to_parquet(challenger_parquet, index=False)
        store.publish(challenger_scores, "challenger", {"run_id": run_id, "snapshot_day": cutoff_day,
                                                        "model_bundle": challenger.bundle_name})
        challenger_summary = {"model_bundle": challenger.bundle_name, "parquet": str(challenger_parquet),
                              "mean_propensity": float(challenger_scores["purchase_probability"].mean())}
    scores_df[crm_cols].to_csv(crm_csv_path, index=False)

    # 6. Drift audit against the champion's own out-of-time reference window (not against
    #    a snapshot of the same day, which would always look stable)
    drift_summary = None
    try:
        reference = build_champion_reference(predictor, builder)
        detector = DataDriftDetector(output_dir=out_dir / "reports")
        drift_report = detector.generate_full_drift_audit(
            reference_df=reference.features,
            current_df=snapshot,
            reference_scores=reference.scores,
            current_scores=scores_df["purchase_probability"].to_numpy(),
            run_id=f"DRIFT_{run_id}",
            target="purchase",
            reference_cutoffs=reference.cutoffs,
            current_cutoff=cutoff_day,
        )
        (monitoring_state or MonitoringState(report_dir=out_dir / "reports")).publish("drift_purchase", drift_report)
        drift_summary = {
            "status": drift_report["overall_status"],
            "drift_ratio": drift_report["feature_drift_summary"]["drift_ratio"],
            "prediction_psi": (drift_report["prediction_drift"] or {}).get("psi"),
            "reference_cutoffs": reference.cutoffs,
            "requires_retraining": drift_report["requires_retraining"],
        }
        logger.info(f"Drift Audit vs reference {reference.cutoffs}: {drift_summary['status']} "
                    f"(Requires Retrain={drift_summary['requires_retraining']})")
    except Exception as e:
        logger.warning(f"Drift Audit could not complete: {e}")

    # 7. Reset Speed Layer Intraday Deltas for fresh cycle
    # 6b. Second model: lapse risk of regular shoppers (skipped if no lapse champion is registered)
    lapse_summary = score_lapse_risk(snapshot, cutoff_day, out_dir, model_path=lapse_model_path)

    speed_proc = SpeedLayerProcessor()
    speed_proc.reset_intraday_deltas()

    duration = (datetime.datetime.now() - start_time).total_seconds()

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "cutoff_day": cutoff_day,
        "model_bundle": predictor.bundle_name,
        "horizon_days": predictor.horizon_days,
        "data_source": data.source,
        "feature_count": len(predictor.features),
        "total_households_scored": len(scores_df),
        "mean_propensity": float(scores_df["purchase_probability"].mean()),
        "top_decile_threshold": float(scores_df[scores_df["decile"] == 10]["purchase_probability"].min()),
        "top_decile_count": int((scores_df["decile"] == 10).sum()),
        "urgent_count": int((scores_df["urgency_tier"] == "URGENT").sum()),
        "lakehouse_records_saved": records_saved,
        "drift_audit": drift_summary,
        "lapse_risk": lapse_summary,
        "challenger": challenger_summary,
        "challenger_bundle": challenger_summary["model_bundle"] if challenger_summary else None,
        "serving_store_backend": store.backend,
        "feature_store": feature_store_summary,
        "duration_seconds": round(duration, 2),
        "artifacts": {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "crm_csv": str(crm_csv_path),
            **({"challenger_parquet": challenger_summary["parquet"]} if challenger_summary else {}),
        },
    }

    manifest_path = out_dir / f"batch_manifest_day_{cutoff_day}.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"=== Enterprise Batch Pipeline Completed in {duration:.2f}s ===")

    return summary


def score_lapse_risk(
    snapshot: pd.DataFrame,
    cutoff_day: int,
    out_dir: Path,
    model_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """Scores eligible regular shoppers with the lapse champion and exports a retention list."""
    path = Path(model_path or (registry_dir_for(LAPSE_TRAINING_CONFIG) / "champion_model.joblib"))
    if not path.exists():
        logger.info(f"No lapse champion at {path}; lapse-risk scoring skipped.")
        return None
    predictor = PropensityPredictor(model_path=path, expected_target="lapse")
    eligible = snapshot[LAPSE.eligible(snapshot)]
    scores = predictor.score_households(eligible)
    csv_path = out_dir / f"lapse_risk_day_{cutoff_day}.csv"
    cols = ["household_key", "lapse_probability", "rank", "decile", "spend_90d",
            "expected_spend_at_risk", "value_rank", "retention_action"]
    scores[cols].sort_values("value_rank").to_csv(csv_path, index=False)
    logger.info(f"Lapse risk scored for {len(scores):,} eligible households -> {csv_path}")
    return {
        "model_bundle": predictor.bundle_name,
        "horizon_days": predictor.horizon_days,
        "eligible_households": int(len(scores)),
        "mean_lapse_probability": float(scores["lapse_probability"].mean()),
        "retention_offer_count": int((scores["retention_action"] == "RETENTION_OFFER").sum()),
        "expected_spend_at_risk_offer_list": float(
            scores.loc[scores["retention_action"] == "RETENTION_OFFER", "expected_spend_at_risk"].sum()
        ),
        "artifact": str(csv_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Enterprise Batch Scoring Pipeline")
    parser.add_argument("--cutoff", type=int, default=711, help="Cutoff snapshot day (default: 711)")
    parser.add_argument("--challenger", default=None, help="Challenger bundle to score for canary traffic")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_batch_pipeline(cutoff_day=args.cutoff, challenger_model_path=args.challenger)
