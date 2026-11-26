"""
Predictor tests: bundle schema validation, legacy (v1) champion compatibility and
serving behaviour on contract-conformant frames.
"""
import joblib
import numpy as np
import pytest

from src.config.settings import settings
from src.features.contract import FEATURE_SETS, FeatureContractError, features_for
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.models.predictor import ModelBundleError, PropensityPredictor


@pytest.fixture(scope="module")
def champion():
    return PropensityPredictor()


def _features(data, predictor, cutoff=150):
    """Features of the predictor's own feature set (the committed champion may be on an older one)."""
    return MultiDomainFeatureBuilder(data, feature_set_version=predictor.feature_set_version).features_at(cutoff)


def test_committed_champion_loads_with_declared_horizon(champion):
    assert champion.feature_set_version in FEATURE_SETS
    assert champion.horizon_days == 14
    assert champion.features == features_for(champion.feature_set_version)


def test_champion_scores_contract_frame(champion, synthetic_data):
    feats = _features(synthetic_data, champion)
    scores = champion.score_households(feats)
    p = scores["purchase_probability"]
    assert ((p >= 0) & (p <= 1)).all()
    assert set(scores["decile"]) == set(range(1, 11))
    assert scores.loc[scores["rank"] == 1, "decile"].item() == 10
    assert set(scores["urgency_tier"]) <= {"URGENT", "REGULAR", "LOW"}


def test_unseen_category_values_do_not_break_scoring(champion, synthetic_data):
    feats = _features(synthetic_data, champion)
    feats["classification_1"] = feats["classification_1"].astype(str)
    feats.loc[feats.index[:3], "classification_1"] = "Age Group 99"
    assert np.isfinite(champion.predict_proba(feats)).all()


def test_missing_feature_is_rejected(champion, synthetic_data):
    feats = _features(synthetic_data, champion).drop(columns=["urgency_ratio"])
    with pytest.raises(FeatureContractError, match="urgency_ratio"):
        champion.predict_proba(feats)


def test_horizon_mismatch_is_rejected():
    with pytest.raises(ModelBundleError, match="horizon"):
        PropensityPredictor(expected_horizon_days=7)


def _legacy_package():
    """A schema-v1 ("bundles" dict) package, as exported from projectphase3, built from the champion."""
    current = joblib.load(settings.MODEL_PATH)
    inner = next(iter(current["bundles"].values())) if "bundles" in current else current
    return {
        "bundles": {"28 multi-domain (projectphase3)": {
            "model": inner["model"],
            "calibrator": inner["calibrator"],
            "features": list(inner["features"]),
            "categorical_features": list(inner["categorical_features"]),
        }},
        "epsilon": 1e-4,
        "prediction_window_days": 14,
    }


def test_legacy_v1_bundle_still_loads(tmp_path, synthetic_data):
    path = tmp_path / "legacy.joblib"
    joblib.dump(_legacy_package(), path)
    legacy = PropensityPredictor(model_path=path)
    assert legacy.bundle["schema_version"] == 1 and legacy.horizon_days == 14
    feats = _features(synthetic_data, legacy)
    np.testing.assert_allclose(legacy.predict_proba(feats), PropensityPredictor().predict_proba(feats))


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
