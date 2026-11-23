"""
Governance tests: base-rate-relative quality gate, paired bootstrap, champion/challenger
non-inferiority, atomic promotion, version history and rollback.
"""
import json

import joblib
import numpy as np
import pytest

from src.config.model_config import GateConfig
from src.pipelines.train_pipeline import run_training_pipeline
from src.training.champion_challenger import compare_with_champion
from src.training.evaluator import ModelEvaluator
from src.training.model_registry import ModelRegistry
from src.training.statistics import paired_bootstrap_delta

# Measured on the real 14-day OOT test set (cutoffs 650/680, base rate 0.7014).
CHAMPION_TEST_METRICS = {
    "roc_auc": 0.8593, "brier_skill_score": 0.3467, "normalized_lift_10": 0.9800,
    "pr_auc_gain": 0.7680, "ece": 0.0193,
}
WEAK_MODEL_TEST_METRICS = {
    "roc_auc": 0.7886, "brier_skill_score": -0.8711, "normalized_lift_10": 0.6972,
    "pr_auc_gain": 0.5574, "ece": 0.4443,
}


# --------------------------------------------------------------------- quality gate
def test_quality_gate_accepts_champion_and_rejects_weak_model(tmp_path):
    registry = ModelRegistry(tmp_path)
    assert registry.evaluate_governance_gate(CHAMPION_TEST_METRICS)["passed"]
    result = registry.evaluate_governance_gate(WEAK_MODEL_TEST_METRICS)
    assert not result["passed"]
    failed = {k for k, c in result["checks"].items() if not c["pass"]}
    assert {"brier_skill_score", "normalized_lift_10", "pr_auc_gain", "ece"} <= failed


def test_gate_is_reachable_at_high_base_rates():
    """The old absolute lift>=1.50 gate was unsatisfiable at a 70% base rate; the new one is not."""
    y = np.array([1] * 700 + [0] * 300)
    perfect = ModelEvaluator.evaluate(y, y * 0.98 + 0.01)
    assert perfect["lift_at_10pct"] < 1.50
    assert perfect["normalized_lift_10"] == pytest.approx(1.0, abs=1e-3)  # metrics are rounded to 4-6 dp


def test_relative_metrics():
    y = np.array([1] * 70 + [0] * 30)
    constant = ModelEvaluator.evaluate(y, np.full(100, 0.7))
    assert constant["brier_skill_score"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------------ bootstrap
def test_paired_bootstrap_identical_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = rng.uniform(0, 1, 400)
    res = paired_bootstrap_delta(y, p, p, n_samples=200)
    assert res["delta"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0


def test_paired_bootstrap_detects_better_model_with_clusters():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    good = np.clip(y * 0.6 + rng.uniform(0, 0.4, 1000), 0, 1)
    noise = rng.uniform(0, 1, 1000)
    groups = np.repeat(np.arange(500), 2)  # each "household" contributes two rows
    res = paired_bootstrap_delta(y, good, noise, groups=groups, n_samples=300)
    assert res["ci_low"] > 0
    assert res["p_a_not_better"] == 0.0


# -------------------------------------------------------------- champion/challenger
@pytest.fixture
def registry_with_champion(synthetic_data, small_training_config, tmp_path):
    registry_dir = tmp_path / "registry"
    first = run_training_pipeline(
        config=small_training_config, data=synthetic_data, output_dir=tmp_path / "run1",
        registry_dir=registry_dir, force_promote=True,
    )
    assert first["promoted"]
    return ModelRegistry(registry_dir), first


def test_identical_retrain_is_non_inferior(registry_with_champion, synthetic_data, small_training_config, tmp_path):
    registry, _ = registry_with_champion
    second = run_training_pipeline(
        config=small_training_config, data=synthetic_data, output_dir=tmp_path / "run2",
        registry_dir=registry.registry_dir, promote=False,
    )
    comparison = second["governance_gate"]["champion_comparison"]
    assert comparison["status"] == "compared"
    assert comparison["passed"] is True
    assert comparison["auc_delta"] == pytest.approx(0.0, abs=1e-9)


def test_worse_challenger_is_blocked(registry_with_champion, synthetic_data, small_training_config):
    from src.features.pit_extractor import MultiDomainFeatureBuilder
    from src.training.dataset_builder import DatasetBuilder

    registry, _ = registry_with_champion
    cfg = small_training_config
    test = DatasetBuilder(MultiDomainFeatureBuilder(synthetic_data, cfg.lookback_days)).build(
        cfg.test_cutoffs, cfg.horizon_days
    )
    random_probs = np.random.default_rng(0).uniform(0, 1, len(test))
    result = compare_with_champion(random_probs, test, registry.champion_path, cfg)
    assert result["status"] == "compared"
    assert result["passed"] is False
    assert result["auc_delta_ci95"][0] < -cfg.gate.non_inferiority_margin_auc


def test_target_change_is_incomparable_and_blocks_promotion(
    registry_with_champion, synthetic_data, small_training_config, tmp_path
):
    registry, first = registry_with_champion
    lenient = GateConfig(min_auc=0, min_brier_skill=-10, min_normalized_lift_10=-10, min_pr_auc_gain=-10, max_ece=1)
    new_target = small_training_config.with_overrides(horizon_days=10, gate=lenient)
    summary = run_training_pipeline(
        config=new_target, data=synthetic_data, output_dir=tmp_path / "run3", registry_dir=registry.registry_dir,
    )
    assert summary["governance_gate"]["quality"]["passed"] is True
    assert summary["governance_gate"]["champion_comparison"]["status"] == "incomparable"
    assert summary["promoted"] is False
    assert joblib.load(registry.champion_path)["horizon_days"] == small_training_config.horizon_days


# ---------------------------------------------------------------- registry / rollback
def _bundle(created_at, name):
    return {"schema_version": 2, "created_at": created_at, "bundle_name": name, "features": [], "horizon_days": 7}


def test_promotion_history_and_rollback(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register_and_promote(_bundle("20260101_000000_000001", "v1"))
    registry.register_and_promote(_bundle("20260201_000000_000001", "v2"))
    assert registry.list_versions() == ["20260101_000000_000001", "20260201_000000_000001"]
    assert registry.current_version() == "20260201_000000_000001"

    rolled = registry.rollback(reason="bad realized AUC")
    assert rolled == "20260101_000000_000001"
    assert joblib.load(registry.champion_path)["bundle_name"] == "v1"
    manifest = json.loads(registry.manifest_path.read_text(encoding="utf-8"))
    assert manifest["activation_event"] == "rollback" and manifest["bundle_name"] == "v1"

    registry.rollback(to_version="20260201_000000_000001")
    assert registry.current_version() == "20260201_000000_000001"

    events = [json.loads(line) for line in registry.audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["promote", "promote", "rollback", "rollback"]
    assert events[2]["reason"] == "bad realized AUC"
    assert not list(tmp_path.rglob("*.tmp"))  # atomic writes leave no temp files


def test_rollback_errors(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register_and_promote(_bundle("20260101_000000_000001", "only"))
    with pytest.raises(ValueError, match="No version older"):
        registry.rollback()
    with pytest.raises(ValueError, match="Unknown version"):
        registry.rollback(to_version="nope")


def test_unregistered_champion_is_archived_before_replacement(tmp_path):
    registry = ModelRegistry(tmp_path)
    joblib.dump({"bundles": {"legacy": {"features": []}}, "prediction_window_days": 14}, registry.champion_path)
    registry.register_and_promote(_bundle("20990101_000000_000001", "new"))

    versions = registry.list_versions()
    assert len(versions) == 2 and versions[0].endswith("_pre_registry")
    registry.rollback()
    restored = joblib.load(registry.champion_path)
    assert "bundles" in restored
    manifest = json.loads(registry.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1 and manifest["horizon_days"] == 14


def test_non_v2_bundle_cannot_be_promoted(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        ModelRegistry(tmp_path).register_and_promote({"schema_version": 1, "created_at": "x"})
