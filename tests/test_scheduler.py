"""
Automated Scheduler and Pipeline Orchestration Unit Tests.
"""
from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.scheduler.daemon import MLSystemScheduler, ScheduledTask
from dags.dag_weekly_batch_propensity import dag as weekly_dag
from dags.dag_monthly_model_retraining import dag as monthly_dag


def test_scheduled_task_interval_trigger():
    mock_func = MagicMock(return_value="OK")
    task = ScheduledTask(
        name="test_interval",
        func=mock_func,
        cron_type="interval",
        interval_seconds=60,
    )

    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)
    # First run without last_run should trigger
    assert task.should_run(t0) is True

    # After execute, last_run is set
    task.execute(as_of=t0)
    assert mock_func.called

    # 30 seconds later: should not trigger
    t1 = t0 + datetime.timedelta(seconds=30)
    assert task.should_run(t1) is False

    # 65 seconds later: should trigger
    t2 = t0 + datetime.timedelta(seconds=65)
    assert task.should_run(t2) is True


def test_scheduled_task_time_matching():
    mock_func = MagicMock(return_value=123)
    # Weekly on Monday (weekday=0) at 02:00
    task = ScheduledTask(
        name="test_weekly",
        func=mock_func,
        cron_type="weekly",
        target_hour=2,
        target_minute=0,
        target_weekday=0,
    )

    # 2026-01-05 is a Monday
    monday_0200 = datetime.datetime(2026, 1, 5, 2, 0, 0)
    assert monday_0200.weekday() == 0
    assert task.should_run(monday_0200) is True

    # Same day, different time: 02:05
    monday_0205 = datetime.datetime(2026, 1, 5, 2, 5, 0)
    assert task.should_run(monday_0205) is False

    # Tuesday at 02:00
    tuesday_0200 = datetime.datetime(2026, 1, 6, 2, 0, 0)
    assert task.should_run(tuesday_0200) is False

    # Monthly on 1st of month at 03:00
    monthly_task = ScheduledTask(
        name="test_monthly",
        func=mock_func,
        cron_type="monthly",
        target_hour=3,
        target_minute=0,
        target_day_of_month=1,
    )

    day1_0300 = datetime.datetime(2026, 2, 1, 3, 0, 0)
    assert monthly_task.should_run(day1_0300) is True

    day2_0300 = datetime.datetime(2026, 2, 2, 3, 0, 0)
    assert monthly_task.should_run(day2_0300) is False


def test_ml_system_scheduler_registration():
    scheduler = MLSystemScheduler()
    # Should have 3 default jobs
    names = [t.name for t in scheduler.tasks]
    assert "weekly_batch_scoring" in names
    assert "monthly_model_retraining" in names
    assert "daily_data_quality_audit" in names


def test_trigger_serving_hot_reload():
    scheduler = MLSystemScheduler(serving_api_url="http://127.0.0.1:8000")

    # Test error handling when serving API is unreachable
    success = scheduler.trigger_serving_hot_reload()
    assert isinstance(success, bool)


def test_scheduler_mocked_execution():
    scheduler = MLSystemScheduler()

    # Mock the internal methods to verify run_once_all behavior
    with patch.object(scheduler, "job_weekly_batch", return_value={"status": "scored"}), \
         patch.object(scheduler, "job_monthly_retraining", return_value={"status": "trained"}), \
         patch.object(scheduler, "job_daily_data_quality", return_value={"status": "PASS"}):

        results = scheduler.run_once_all()
        assert results["weekly_batch_scoring"]["status"] == "SUCCESS"
        assert results["monthly_model_retraining"]["status"] == "SUCCESS"
        assert results["daily_data_quality_audit"]["status"] == "SUCCESS"


def test_airflow_dags_structure():
    # Verify weekly batch DAG configuration
    assert weekly_dag.dag_id == "weekly_batch_propensity_scoring"
    assert weekly_dag.schedule_interval == "0 2 * * 1"

    # Verify monthly retraining DAG configuration
    assert monthly_dag.dag_id == "monthly_propensity_model_retraining"
    assert monthly_dag.schedule_interval == "0 3 1 * *"
