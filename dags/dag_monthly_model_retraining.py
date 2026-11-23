"""
Airflow DAG: Monthly Model Retraining, Governance Quality Gates & Zero-Downtime Hot-Reload.
Schedule: 1st of every month at 03:00 UTC (0 3 1 * *).
Orchestrates:
  1. Multi-Cutoff Point-in-Time Temporal Dataset Construction.
  2. Champion Model (LightGBM) Training & calibration chosen on validation (none/Platt/Isotonic).
  3. Governance Gate Verification on the out-of-time test split (thresholds: src/config/model_config.py).
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


def task_train_and_evaluate(target: str = "purchase", **kwargs) -> Dict[str, Any]:
    """Retrains one target's champion candidate and applies the governance gate."""
    from src.config.model_config import TRAINING_CONFIGS
    from src.pipelines.train_pipeline import run_training_pipeline

    logger.info(f"Starting automated retraining for target '{target}'...")
    # Target, splits and hyper-parameters come from src/config/model_config.py.
    # Automated runs never override the gate: force_promote is a manual, audited action.
    summary = run_training_pipeline(config=TRAINING_CONFIGS[target], force_promote=False)
    test_metrics = summary["test_metrics"]
    gate = summary["governance_gate"]
    if not gate["passed"]:
        failed_quality = [k for k, c in gate["quality"]["checks"].items() if not c["pass"]]
        comparison = gate["champion_comparison"]
        logger.warning(
            f"Candidate blocked. Failed quality checks: {failed_quality or 'none'}; "
            f"champion comparison: {comparison['status']} "
            f"(dAUC CI95={comparison.get('auc_delta_ci95')}, reason={comparison.get('reason')})"
        )
    logger.info(
        f"Retraining finished. Promoted: {summary['promoted']}. "
        f"OOT test ROC-AUC: {test_metrics['roc_auc']:.4f}, "
        f"Brier: {test_metrics['brier_score']:.4f}, "
        f"Top Decile Lift: {test_metrics['top_decile_lift']:.2f}x"
    )
    ti = kwargs.get("ti")
    if ti is not None:
        ti.xcom_push(key="is_promoted", value=summary["promoted"])
        ti.xcom_push(key="model_path", value=summary.get("promoted_path"))
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
        op_kwargs={"target": "purchase"},
        provide_context=True,
    )

    # Second model (lapse risk). It is not served by the online API, so no hot-reload is needed.
    retrain_lapse_task = PythonOperator(
        task_id="retrain_and_evaluate_lapse_candidate",
        python_callable=task_train_and_evaluate,
        op_kwargs={"target": "lapse"},
        provide_context=True,
    )

    hot_reload_task = PythonOperator(
        task_id="zero_downtime_serving_hot_reload",
        python_callable=task_notify_and_hot_reload,
        provide_context=True,
    )

    retrain_task >> hot_reload_task
