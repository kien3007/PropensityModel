"""
Training stack tests on the synthetic dataset: dataset builder, calibrators, evaluator,
registry and the end-to-end retraining pipeline.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config.model_config import TrainingConfig
from src.features.contract import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.models.predictor import PropensityPredictor
from src.pipelines.train_pipeline import run_training_pipeline
from src.training.calibrator import IsotonicCalibrator, PlattCalibrator, select_calibrator
from src.training.dataset_builder import DatasetBuilder
from src.training.evaluator import ModelEvaluator, expected_calibration_error
from src.training.model_registry import ModelRegistry


def test_labels_use_strict_future_window(synthetic_data):
    builder = MultiDomainFeatureBuilder(synthetic_data, lookback_days=60)
    cutoff, horizon = 100, 7
    labels = builder.labels_at(cutoff, horizon)
    b = builder.baskets
    expected = set(b[(b["DAY"] > cutoff) & (b["DAY"] <= cutoff + horizon)]["household_key"])
    assert set(labels.loc[labels["target"] == 1, "household_key"]) == expected
    assert len(labels) == len(synthetic_data.households)


def test_returned_basket_is_not_a_purchase(synthetic_tables, synthetic_data):
    builder = MultiDomainFeatureBuilder(synthetic_data)
    assert 1 not in set(builder.baskets["BASKET_ID"])  # the all-returns basket (spend <= 0)


def test_dataset_builder_rejects_incomplete_label_window(synthetic_data):
    builder = DatasetBuilder(MultiDomainFeatureBuilder(synthetic_data))
    last_day = int(synthetic_data.transactions["DAY"].max())
    with pytest.raises(ValueError, match="exceeds last observed DAY"):
        builder.build_snapshot(cutoff=last_day - 3, horizon_days=7)


def test_dataset_builder_splits_keep_categoricals(synthetic_data, small_training_config):
    splits = DatasetBuilder(MultiDomainFeatureBuilder(synthetic_data, 60)).build_splits(small_training_config)
    for split in splits.values():
        assert list(split.X.columns) == MODEL_FEATURES
        for col in CATEGORICAL_FEATURES:
            assert split.X[col].dtype.name == "category"
    n_hh = len(synthetic_data.households)
    assert len(splits["fit"].frame) == n_hh * len(small_training_config.fit_cutoffs)


def test_config_rejects_overlapping_splits():
    with pytest.raises(ValueError, match="Temporal leakage"):
        TrainingConfig(horizon_days=14, fit_cutoffs=(100, 200), val_cutoffs=(210,), test_cutoffs=(300,))


def test_calibrators_and_selection():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < raw ** 2).astype(int)  # raw scores are over-confident
    for cal in (PlattCalibrator(), IsotonicCalibrator()):
        p = cal.fit(raw, y).predict_proba(raw)
        assert ((p >= 0) & (p <= 1)).all()
        assert expected_calibration_error(y, p) < expected_calibration_error(y, raw)

    chosen, report = select_calibrator(raw, y, n_folds=3)
    assert set(report) == {"none", "platt", "isotonic"}
    assert chosen.method != "none"


def test_evaluator_reports_lift_ceiling():
    y = np.array([1] * 70 + [0] * 30)
    p = np.linspace(1, 0, 100)
    m = ModelEvaluator.evaluate(y, p)
    assert m["max_attainable_lift_10"] == pytest.approx(1 / 0.7, rel=1e-3)
    assert m["lift_at_10pct"] <= m["max_attainable_lift_10"]
    assert {"roc_auc", "pr_auc", "brier_score", "ece", "top_decile_lift"} <= set(m)


def test_governance_gate(tmp_path):
    registry = ModelRegistry(registry_dir=tmp_path)
    assert registry.evaluate_governance_gate({"roc_auc": 0.85, "brier_score": 0.16, "lift_at_10pct": 1.8})["passed"]
    assert not registry.evaluate_governance_gate({"roc_auc": 0.6, "brier_score": 0.25, "lift_at_10pct": 1.1})["passed"]


def test_end_to_end_training_and_promotion(synthetic_data, small_training_config, tmp_path):
    registry_dir = tmp_path / "registry"
    summary = run_training_pipeline(
        config=small_training_config,
        data=synthetic_data,
        output_dir=tmp_path / "out",
        registry_dir=registry_dir,
        force_promote=True,
    )

    assert summary["promoted"] is True
    assert summary["test_metrics"]["roc_auc"] > 0.6
    assert summary["calibration_method"] in {"none", "platt", "isotonic"}
    assert set(summary["baselines"]) == {"logistic_regression", "recency_rule"}

    bundle = joblib.load(registry_dir / "champion_model.joblib")
    assert bundle["schema_version"] == 2
    assert bundle["horizon_days"] == small_training_config.horizon_days
    assert bundle["features"] == MODEL_FEATURES
    assert bundle["data_fingerprint"] == synthetic_data.fingerprint()
    assert list(bundle["test_cutoffs"]) == list(small_training_config.test_cutoffs)
    assert "source_project" not in bundle

    manifest = json.loads((registry_dir / "latest_model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["horizon_days"] == small_training_config.horizon_days
    assert manifest["git_sha"]

    # The promoted bundle is servable and reproduces the pipeline's own test predictions.
    predictor = PropensityPredictor(model_path=registry_dir / "champion_model.joblib")
    builder = MultiDomainFeatureBuilder(synthetic_data, lookback_days=small_training_config.lookback_days)
    test = DatasetBuilder(builder).build(small_training_config.test_cutoffs, small_training_config.horizon_days)
    served = ModelEvaluator.evaluate(test["target"], predictor.predict_proba(test))
    assert served["roc_auc"] == pytest.approx(summary["test_metrics"]["roc_auc"], abs=1e-9)


def test_gate_failure_blocks_promotion(synthetic_data, small_training_config, tmp_path):
    strict = small_training_config.with_overrides(
        gate=type(small_training_config.gate)(min_auc=0.999, max_brier=0.0, min_lift_10=99.0)
    )
    summary = run_training_pipeline(
        config=strict, data=synthetic_data, output_dir=tmp_path / "out", registry_dir=tmp_path / "registry",
    )
    assert summary["promoted"] is False
    assert not (tmp_path / "registry" / "champion_model.joblib").exists()
    assert pd.io.common.file_exists(summary["candidate_path"])


def test_no_promote_never_touches_registry(synthetic_data, small_training_config, tmp_path):
    summary = run_training_pipeline(
        config=small_training_config, data=synthetic_data, output_dir=tmp_path / "out",
        registry_dir=tmp_path / "registry", promote=False, force_promote=True,
    )
    assert summary["promoted"] is False
    assert not (tmp_path / "registry" / "champion_model.joblib").exists()
