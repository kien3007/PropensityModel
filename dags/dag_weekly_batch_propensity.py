"""
Airflow DAG: Weekly Batch Propensity Scoring & Lakehouse Materialization.
Schedule: Every Monday at 02:00 UTC (0 2 * * 1).
Orchestrates:
  1. Lakehouse Data Quality & Anomaly Detection Audit.
  2. Master Dataset Bronze -> Silver -> Gold Ingestion.
  3. Point-in-Time 28-feature Snapshot Extraction.
  4. Champion LightGBM Inference & Isotonic Calibration.
  5. Multi-channel CRM Segment Export & Redis Cache Invalidation.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any, Dict

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Airflow requires POSIX/Linux (Docker, Cloud Composer, MWAA) due to fcntl
    AIRFLOW_AVAILABLE = False

    class DAG:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.dag_id = args[0] if args else kwargs.get("dag_id", "")
            self.schedule_interval = kwargs.get("schedule_interval")
            self.default_args = kwargs.get("default_args", {})
            self.tags = kwargs.get("tags", [])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class PythonOperator:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.task_id = kwargs.get("task_id", "")
            self.python_callable = kwargs.get("python_callable")
            self.op_kwargs = kwargs.get("op_kwargs", {})
        def __rshift__(self, other):
            return other

logger = logging.getLogger("airflow.task")

default_args = {
    "owner": "mlops-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": datetime.timedelta(minutes=5),
}


def task_lakehouse_data_quality_audit(**kwargs) -> Dict[str, Any]:
    """Validates Lakehouse data freshness and schema integrity."""
    from src.lakehouse.engine import LakehouseEngine

    logger.info("Starting Lakehouse Data Quality Framework verification...")
    engine = LakehouseEngine()
    result = engine.run_data_quality_framework()
    if result["status"] != "PASS":
        raise ValueError(f"Data Quality Audit FAILED: {result['failures']}")
    logger.info(f"Data Quality Audit PASSED: {result['total_assertions']} checks satisfied.")
    return result


def task_execute_batch_scoring(cutoff_day: int = 711, **kwargs) -> Dict[str, Any]:
    """Executes the enterprise weekly batch scoring pipeline."""
    from src.pipelines.batch_pipeline import run_batch_pipeline

    logger.info(f"Executing Batch Scoring Pipeline for Cutoff Day {cutoff_day}...")
    summary = run_batch_pipeline(cutoff_day=cutoff_day)
    logger.info(
        f"Batch Scoring Complete: {summary['total_households_scored']} households scored. "
        f"Mean Propensity: {summary['mean_propensity']:.4f}."
    )
    return summary


def task_sync_serving_cache(**kwargs) -> None:
    """Warms Redis speed layer cache and resets intraday delta streams."""
    from src.speed_layer.delta_stream import SpeedLayerProcessor

    logger.info("Resetting intraday delta streams for fresh weekly scoring cycle...")
    processor = SpeedLayerProcessor()
    cleared = processor.reset_intraday_deltas()
    logger.info(f"Speed layer state initialized. Intraday deltas reset ({cleared} entries cleared).")


with DAG(
    dag_id="weekly_batch_propensity_scoring",
    default_args=default_args,
    description="Weekly Batch Propensity Scoring, Lakehouse Gold Sync, and CRM Audience Export",
    schedule_interval="0 2 * * 1",  # Every Monday at 02:00 AM UTC
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["propensity", "batch-scoring", "lakehouse", "lambda-architecture"],
) as dag:

    dq_check = PythonOperator(
        task_id="lakehouse_data_quality_audit",
        python_callable=task_lakehouse_data_quality_audit,
    )

    batch_scoring = PythonOperator(
        task_id="execute_batch_scoring",
        python_callable=task_execute_batch_scoring,
        op_kwargs={"cutoff_day": 711},
    )

    serving_sync = PythonOperator(
        task_id="sync_serving_cache",
        python_callable=task_sync_serving_cache,
    )

    dq_check >> batch_scoring >> serving_sync
