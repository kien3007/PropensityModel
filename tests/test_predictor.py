"""
Predictor tests: bundle schema validation, legacy (v1) champion compatibility and
serving behaviour on contract-conformant frames.
"""
import joblib
import numpy as np
import pytest

from src.config.settings import settings
from src.features.contract import FeatureContractError
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.models.predictor import ModelBundleError, PropensityPredictor


@pytest.fixture(scope="module")
def champion():
    return PropensityPredictor()


def test_committed_champion_loads_with_declared_horizon(champion):
    assert champion.feature_set_version == "multidomain_v1"
    assert champion.horizon_days == 14
    assert len(champion.features) == 28


def test_champion_scores_contract_frame(champion, synthetic_data):
    feats = MultiDomainFeatureBuilder(synthetic_data).features_at(150)
    scores = champion.score_households(feats)
    p = scores["purchase_probability"]
    assert ((p >= 0) & (p <= 1)).all()
    assert set(scores["decile"]) == set(range(1, 11))
    assert scores.loc[scores["rank"] == 1, "decile"].item() == 10
    assert set(scores["urgency_tier"]) <= {"URGENT", "REGULAR", "LOW"}


def test_unseen_category_values_do_not_break_scoring(champion, synthetic_data):
    feats = MultiDomainFeatureBuilder(synthetic_data).features_at(150)
    feats["classification_1"] = feats["classification_1"].astype(str)
    feats.loc[feats.index[:3], "classification_1"] = "Age Group 99"
    assert np.isfinite(champion.predict_proba(feats)).all()


def test_missing_feature_is_rejected(champion, synthetic_data):
    feats = MultiDomainFeatureBuilder(synthetic_data).features_at(150).drop(columns=["urgency_ratio"])
    with pytest.raises(FeatureContractError, match="urgency_ratio"):
        champion.predict_proba(feats)


def test_horizon_mismatch_is_rejected():
    with pytest.raises(ModelBundleError, match="horizon"):
        PropensityPredictor(expected_horizon_days=7)


def _legacy_package():
    return joblib.load(settings.MODEL_PATH)


def test_unknown_feature_set_is_rejected(tmp_path):
    pkg = _legacy_package()
    inner = next(iter(pkg["bundles"].values()))
    inner["features"] = inner["features"][:-1]
    path = tmp_path / "bad.joblib"
    joblib.dump(pkg, path)
    with pytest.raises(ModelBundleError, match="does not match any supported feature set"):
        PropensityPredictor(model_path=path)


def test_ambiguous_legacy_bundle_is_rejected(tmp_path):
    pkg = _legacy_package()
    inner = next(iter(pkg["bundles"].values()))
    pkg["bundles"]["another"] = inner
    path = tmp_path / "two.joblib"
    joblib.dump(pkg, path)
    with pytest.raises(ModelBundleError, match="exactly one"):
        PropensityPredictor(model_path=path)


def test_bundle_without_horizon_is_rejected(tmp_path):
    pkg = _legacy_package()
    pkg.pop("prediction_window_days")
    path = tmp_path / "nohorizon.joblib"
    joblib.dump(pkg, path)
    with pytest.raises(ModelBundleError, match="horizon"):
        PropensityPredictor(model_path=path)
