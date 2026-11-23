"""
Target definitions (purchase / lapse), lapse model training and value-ranked retention scoring.
"""
import joblib
import pytest

from src.config.model_config import LAPSE_TRAINING_CONFIG, TRAINING_CONFIGS, GateConfig, TrainingConfig
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.features.targets import LAPSE, PURCHASE, get_target
from src.models.predictor import ModelBundleError, PropensityPredictor
from src.pipelines.batch_pipeline import score_lapse_risk
from src.pipelines.train_pipeline import run_training_pipeline
from src.training.dataset_builder import DatasetBuilder

LENIENT_GATE = GateConfig(min_auc=0, min_brier_skill=-10, min_normalized_lift_10=None, min_lift_10=None,
                          min_pr_auc_gain=-10, max_ece=1)


def test_lapse_is_inverse_purchase_on_regular_shoppers(synthetic_data):
    builder = MultiDomainFeatureBuilder(synthetic_data)
    cutoff, horizon = 120, 14
    feats = builder.features_at(cutoff)
    eligible = feats[LAPSE.eligible(feats)]
    assert 0 < len(eligible) < len(feats)
    assert (eligible["baskets_count_30d"] >= 1).all() and (eligible["baskets_count_90d"] >= 4).all()

    purchase = PURCHASE.labels(builder, cutoff, horizon).set_index("household_key")["target"]
    lapse = LAPSE.labels(builder, cutoff, horizon).set_index("household_key")["target"]
    assert (lapse == 1 - purchase).all()


def test_dataset_builder_keeps_only_eligible_rows(synthetic_data):
    builder = MultiDomainFeatureBuilder(synthetic_data)
    rows = DatasetBuilder(builder).build([120, 140], horizon_days=14, target=LAPSE)
    assert (rows["baskets_count_90d"] >= 4).all()
    assert rows["target"].between(0, 1).all()


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="Unknown target"):
        get_target("churn")
    with pytest.raises(ValueError, match="Unknown target"):
        TrainingConfig(target="churn")


def test_lapse_config_is_leakage_safe_and_registered():
    assert TRAINING_CONFIGS["lapse"] is LAPSE_TRAINING_CONFIG
    cfg = LAPSE_TRAINING_CONFIG
    assert max(cfg.val_cutoffs) + cfg.horizon_days <= min(cfg.test_cutoffs)
    assert cfg.gate.min_normalized_lift_10 is None and cfg.gate.min_lift_10 is not None


@pytest.fixture
def lapse_config(small_training_config):
    return small_training_config.with_overrides(target="lapse", registry_subdir="lapse", gate=LENIENT_GATE)


def test_lapse_training_end_to_end_and_value_scoring(synthetic_data, lapse_config, tmp_path):
    summary = run_training_pipeline(
        config=lapse_config, data=synthetic_data, output_dir=tmp_path / "out", registry_dir=tmp_path / "reg",
    )
    assert summary["promoted"] and summary["target"] == "lapse"
    bundle = joblib.load(tmp_path / "reg" / "champion_model.joblib")
    assert bundle["target"] == "lapse"
    assert set(summary["governance_gate"]["quality"]["checks"]) == {"roc_auc", "brier_skill_score", "pr_auc_gain", "ece"}

    snapshot = MultiDomainFeatureBuilder(synthetic_data).features_at(180)
    result = score_lapse_risk(snapshot, 180, tmp_path, model_path=tmp_path / "reg" / "champion_model.joblib")
    assert result["eligible_households"] == int(LAPSE.eligible(snapshot).sum())
    exported = __import__("pandas").read_csv(result["artifact"])
    assert exported["value_rank"].is_monotonic_increasing
    top = exported[exported["retention_action"] == "RETENTION_OFFER"]
    assert top["expected_spend_at_risk"].min() >= exported.loc[
        exported["retention_action"] != "RETENTION_OFFER", "expected_spend_at_risk"].max()


def test_purchase_champion_cannot_be_compared_with_lapse_challenger(
    synthetic_data, small_training_config, lapse_config, tmp_path
):
    registry = tmp_path / "shared"
    run_training_pipeline(config=small_training_config, data=synthetic_data, output_dir=tmp_path / "a",
                          registry_dir=registry, force_promote=True)
    summary = run_training_pipeline(config=lapse_config, data=synthetic_data, output_dir=tmp_path / "b",
                                    registry_dir=registry)
    comparison = summary["governance_gate"]["champion_comparison"]
    assert comparison["status"] == "incomparable" and "purchase" in comparison["reason"]
    assert summary["promoted"] is False


def test_predictor_enforces_expected_target(tmp_path):
    with pytest.raises(ModelBundleError, match="target"):
        PropensityPredictor(expected_target="lapse")  # the committed champion predicts purchase


def test_lapse_scoring_skipped_without_champion(synthetic_data, tmp_path):
    snapshot = MultiDomainFeatureBuilder(synthetic_data).features_at(150)
    assert score_lapse_risk(snapshot, 150, tmp_path, model_path=tmp_path / "missing.joblib") is None
