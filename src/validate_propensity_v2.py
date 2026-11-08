"""Validate V2 artifacts, temporal contract, and deployable score output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    model_dir = Path("artifacts/propensity_v2/model")
    audit_dir = Path("artifacts/propensity_v2/data_understanding")
    required = [
        model_dir / "metrics_v2.json",
        model_dir / "rolling_backtest_metrics.csv",
        model_dir / "bootstrap_confidence_intervals.csv",
        model_dir / "top_k_business_metrics.csv",
        model_dir / "test_predictions.csv",
        model_dir / "latest_scores_v2.csv",
        model_dir / "grouped_feature_importance.csv",
        model_dir / "feature_drift_psi.csv",
        audit_dir / "data_audit.json",
        audit_dir / "target_horizon_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing V2 artifacts: {missing}")

    metrics = json.loads((model_dir / "metrics_v2.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(model_dir / "test_predictions.csv")
    scores = pd.read_csv(model_dir / "latest_scores_v2.csv")
    backtest = pd.read_csv(model_dir / "rolling_backtest_metrics.csv")
    intervals = pd.read_csv(model_dir / "bootstrap_confidence_intervals.csv")
    top_k = pd.read_csv(model_dir / "top_k_business_metrics.csv")

    target = "target_purchase_next_7d"
    expected_test_rows = int(metrics["test_metrics"]["rows"])
    assert len(predictions) == expected_test_rows
    assert not predictions.duplicated(["household_key", "snapshot_day"]).any()
    assert set(predictions[target].unique()).issubset({0, 1})
    assert predictions["purchase_probability"].between(0, 1).all()

    assert scores["household_key"].is_unique
    assert scores["purchase_probability_next_7d"].between(0, 1).all()
    assert scores["rank"].is_unique
    assert sorted(scores["rank"].tolist()) == list(range(1, len(scores) + 1))
    assert scores["percentile"].between(0, 1).all()

    split = metrics["split_manifest"]
    assert split["train"]["snapshot_max"] < split["embargo_days"][0]
    assert split["embargo_days"][0] < split["validation"]["snapshot_min"]
    assert split["validation"]["snapshot_max"] < split["embargo_days"][1]
    assert split["embargo_days"][1] < split["test"]["snapshot_min"]

    assert backtest.groupby("model")["fold"].nunique().min() == 3
    assert set(intervals["metric"]) == {
        "pr_auc",
        "roc_auc",
        "brier_score",
        "lift_at_top_10pct",
    }
    assert (intervals["ci_lower_95"] <= intervals["estimate_mean"]).all()
    assert (intervals["estimate_mean"] <= intervals["ci_upper_95"]).all()
    assert top_k["population_fraction"].is_monotonic_increasing
    assert top_k.iloc[-1]["population_fraction"] == 1.0

    validation = {
        "status": "PASS",
        "selected_model": metrics["selected_model"],
        "test_rows": len(predictions),
        "score_rows": len(scores),
        "score_snapshot": int(scores["snapshot_day"].iloc[0]),
        "rolling_models": int(backtest["model"].nunique()),
        "rolling_folds": int(backtest["fold"].nunique()),
        "test_pr_auc": metrics["test_metrics"]["pr_auc"],
        "test_brier_score": metrics["test_metrics"]["brier_score"],
        "lift_at_top_10pct": metrics["test_metrics"]["lift_at_top_10pct"],
    }
    output = model_dir / "validation_report.json"
    output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
