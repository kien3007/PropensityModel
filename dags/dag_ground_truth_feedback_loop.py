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


def task_evaluate_ground_truth_attribution(cutoff_day: int = 690, horizon_days: int = 14, **kwargs) -> Dict[str, Any]:
    """Runs automated ground-truth join and attribution analysis."""
    from src.monitoring.feedback_loop import GroundTruthFeedbackLoop

    logger.info(f"Running Ground-Truth Feedback Attribution for Cutoff Day {cutoff_day}...")
    loop = GroundTruthFeedbackLoop()
    result = loop.evaluate_realized_performance(cutoff_day=cutoff_day, horizon_days=horizon_days)

    logger.info(
        f"Attribution Result: Status={result['attribution_status']} | "
        f"Realized AUC={result['realized_roc_auc']:.4f} | "
        f"Realized Lift={result['realized_top_decile_lift']:.2f}x"
    )

    if result["attribution_status"] == "CRITICAL_DRIFT":
        raise ValueError(f"CRITICAL: Realized AUC dropped to {result['realized_roc_auc']:.4f}. Retraining urgent!")

    return result


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
        op_kwargs={"cutoff_day": 690, "horizon_days": 14},
    )
