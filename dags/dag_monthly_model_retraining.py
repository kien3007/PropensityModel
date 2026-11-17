"""
Airflow DAG: Monthly Model Retraining, Governance Quality Gates & Zero-Downtime Hot-Reload.
Schedule: 1st of every month at 03:00 UTC (0 3 1 * *).
Orchestrates:
  1. Multi-Cutoff Point-in-Time Temporal Dataset Construction.
  2. Champion Model (LightGBM) Training & Platt/Isotonic Calibration.
  3. Governance Gate Verification (ROC-AUC >= 0.85, Top Decile Lift >= 1.30, Brier <= 0.16).
  4. Champion Registry Promotion & Lineage Tracking.
  5. Zero-Downtime Serving Gateway Hot-Reload via REST Webhook.
"""
from __future__ import annotations

import datetime
import json
import logging
import urllib.request
from typing import Any, Dict

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
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
            self.provide_context = kwargs.get("provide_context", False)
        def __rshift__(self, other):
            return other

logger = logging.getLogger("airflow.task")

default_args = {
    "owner": "mlops-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=10),
}


def task_train_and_evaluate(**kwargs) -> Dict[str, Any]:
    """Executes multi-cutoff temporal training and calibration pipeline."""
    from src.pipelines.train_pipeline import run_training_pipeline

    logger.info("Starting automated model retraining pipeline...")
    summary = run_training_pipeline(
        train_cutoffs=[250, 300, 350, 400, 450],
        val_cutoff=537,
        horizon_days=7,
        force_promote=False,
    )
    logger.info(
        f"Retraining finished. Promoted: {summary['promoted']}. "
        f"Validation ROC-AUC: {summary['calibrated_metrics']['roc_auc']:.4f}, "
        f"Brier: {summary['calibrated_metrics']['brier_score']:.4f}, "
        f"Top Decile Lift: {summary['calibrated_metrics']['top_decile_lift']:.2f}x"
    )
    ti = kwargs.get("ti")
    if ti is not None:
        ti.xcom_push(key="is_promoted", value=summary["promoted"])
        ti.xcom_push(key="model_path", value=summary.get("promoted_model_path"))
    return summary


def task_notify_and_hot_reload(**kwargs) -> None:
    """Invokes serving gateway /v1/models/reload endpoint to swap active weights without downtime."""
    from src.config.settings import settings

    ti = kwargs.get("ti")
    is_promoted = True
    if ti is not None:
        is_promoted = ti.xcom_pull(key="is_promoted", task_ids="retrain_and_evaluate_candidate")

    if not is_promoted:
        logger.info("Candidate did not satisfy governance gates. Serving API remains on current champion model.")
        return

    reload_url = f"http://{settings.APP_HOST}:{settings.APP_PORT}/v1/models/reload"
    logger.info(f"Triggering Zero-Downtime Hot-Reload webhook: {reload_url}...")
    try:
        req = urllib.request.Request(
            reload_url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            res_data = json.loads(resp.read().decode())
            logger.info(f"Serving Gateway successfully hot-reloaded: {res_data}")
    except Exception as e:
        logger.warning(f"Could not reach Serving Gateway for webhook hot-reload ({e}). Scheduled reload will occur on next pod restart.")


with DAG(
    dag_id="monthly_propensity_model_retraining",
    default_args=default_args,
    description="Monthly Propensity Model Retraining, Governance Quality Gates & Hot-Reload",
    schedule_interval="0 3 1 * *",  # 1st of every month at 03:00 AM UTC
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["propensity", "retraining", "mlops", "governance"],
) as dag:

    retrain_task = PythonOperator(
        task_id="retrain_and_evaluate_candidate",
        python_callable=task_train_and_evaluate,
        provide_context=True,
    )

    hot_reload_task = PythonOperator(
        task_id="zero_downtime_serving_hot_reload",
        python_callable=task_notify_and_hot_reload,
        provide_context=True,
    )

    retrain_task >> hot_reload_task
