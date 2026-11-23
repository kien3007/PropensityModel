"""
Airflow DAG: Ground-Truth Feedback Loop & Prospective Accuracy Attribution.
Schedule: Every Wednesday at 04:00 UTC (0 4 * * 3).
Orchestrates:
  1. Identifies matured prediction cohorts (matured past 14-day prospective window).
  2. Joins baseline predictions with realized actual transaction data.
  3. Calculates Realized ROC-AUC, Brier score, and Top Decile Lift.
  4. Stores attribution results into Gold Lakehouse table fct_model_attribution.
  5. Alerts if realized accuracy drops below SLA (AUC < 0.70).
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
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


def task_evaluate_ground_truth_attribution(**kwargs) -> Dict[str, Any]:
    """
    Evaluates every target's champion on its latest matured cutoff (horizon from the bundle)
    and retrains when realized AUC dropped more than REALIZED_AUC_DROP_ALERT below the
    accepted AUC. Promotion of the retrained candidate still goes through the governance gate.
    """
    from src.scheduler.daemon import MLSystemScheduler

    reports = MLSystemScheduler().job_ground_truth_feedback()
    for target, r in reports.items():
        if target in ("purchase", "lapse"):
            logger.info(f"[{target}] realized AUC {r['realized_roc_auc']:.4f} vs accepted {r['expected_roc_auc']:.4f} "
                        f"-> {r['attribution_status']}")
    logger.info(f"Retraining decision: {reports['retraining_decision']}")
    return reports


with DAG(
    dag_id="weekly_ground_truth_feedback_loop",
    default_args=default_args,
    description="Automated Prospective Accuracy Attribution & Feedback Loop",
    schedule_interval="0 4 * * 3",  # Every Wednesday at 04:00 AM UTC
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["propensity", "feedback-loop", "attribution", "mlops"],
) as dag:

    attribution_task = PythonOperator(
        task_id="evaluate_realized_accuracy",
        python_callable=task_evaluate_ground_truth_attribution,
    )
