"""
Unit and Integration tests for Model Training, Calibration, Evaluation, and Registry Gates.
Tests DatasetBuilder, ModelTrainer, PlattCalibrator, ModelEvaluator, and ModelRegistry.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.pit_extractor import MODEL_FEATURES
from src.training.dataset_builder import DatasetBuilder
from src.training.trainer import ModelTrainer
from src.training.calibrator import PlattCalibrator
from src.training.evaluator import ModelEvaluator
from src.training.model_registry import ModelRegistry


@pytest.fixture
def synthetic_transactions():
    """Generates synthetic transactions for quick deterministic testing."""
    return pd.DataFrame({
        "household_key": [1, 1, 2, 2, 3],
        "BASKET_ID": [101, 102, 201, 202, 301],
        "DAY": [10, 50, 20, 105, 102],
        "SALES_VALUE": [15.0, 25.0, 30.0, 50.0, 10.0],
    })


def test_dataset_builder_target_labeling(synthetic_transactions):
    cutoff = 100
    horizon = 7

    # Household 2 has purchase on day 105 (in 100-107] -> target = 1
    # Household 3 has purchase on day 102 (in 100-107] -> target = 1
    # Household 1 has no purchase in 100-107 -> target = 0
    snapshot = DatasetBuilder.build_snapshot_with_target(
        transactions=synthetic_transactions,
        cutoff_day=cutoff,
        horizon_days=horizon,
        feature_cols=MODEL_FEATURES,
    )

    assert len(snapshot) == 2500
    assert "target" in snapshot.columns
    assert snapshot.loc[snapshot["household_key"] == 2, "target"].iloc[0] == 1
    assert snapshot.loc[snapshot["household_key"] == 3, "target"].iloc[0] == 1
    assert snapshot.loc[snapshot["household_key"] == 1, "target"].iloc[0] == 0


def test_model_trainer_and_calibrator():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
    y = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

    trainer = ModelTrainer(
        features=MODEL_FEATURES,
        lgbm_params={"n_estimators": 10, "num_leaves": 7, "min_child_samples": 5, "random_state": 42, "verbosity": -1},
    )
    model = trainer.train_lightgbm(X, y)
    raw_probs = model.predict_proba(X[MODEL_FEATURES])[:, 1]
    assert len(raw_probs) == n
    assert ((raw_probs >= 0.0) & (raw_probs <= 1.0)).all()

    # Feature importances
    fi_df = trainer.get_feature_importances(model)
    assert len(fi_df) == len(MODEL_FEATURES)
    assert "feature" in fi_df.columns and "importance" in fi_df.columns

    # Platt Calibration
    calibrator = PlattCalibrator()
    calibrator.fit(raw_probs, y)
    calibrated_probs = calibrator.predict_proba(raw_probs)
    assert len(calibrated_probs) == n
    assert ((calibrated_probs >= 0.0) & (calibrated_probs <= 1.0)).all()

    # Evaluation
    metrics = ModelEvaluator.evaluate(y, calibrated_probs)
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "lift_at_10pct" in metrics
    assert metrics["brier_score"] >= 0.0


def test_model_registry_governance_gate(tmp_path):
    registry = ModelRegistry(registry_dir=tmp_path)

    # Test gate passing
    passing_metrics = {"roc_auc": 0.85, "brier_score": 0.16, "lift_at_10pct": 1.80}
    gate_pass = registry.evaluate_governance_gate(passing_metrics)
    assert gate_pass["passed"] is True

    # Test gate failing
    failing_metrics = {"roc_auc": 0.60, "brier_score": 0.25, "lift_at_10pct": 1.10}
    gate_fail = registry.evaluate_governance_gate(failing_metrics)
    assert gate_fail["passed"] is False

    # Test model packaging & promotion
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
    y = np.random.choice([0, 1], size=50)
    trainer = ModelTrainer(features=MODEL_FEATURES, lgbm_params={"n_estimators": 5, "min_child_samples": 2, "verbosity": -1})
    model = trainer.train_lightgbm(X, y)

    promoted_file = registry.register_and_promote(
        model=model,
        features=MODEL_FEATURES,
        metrics=passing_metrics,
        fit_cutoffs=[300, 400],
        val_cutoffs=[500],
    )
    assert promoted_file.exists()
    assert (tmp_path / "latest_model_manifest.json").exists()
