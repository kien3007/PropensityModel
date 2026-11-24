"""
Feature store observability (Evidently): snapshot profiles, injected-shift detection, trends,
batch publication to MonitoringState and the Prometheus gauges.
"""
from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

pytest.importorskip("evidently")

from src.config.settings import settings  # noqa: E402
from src.feature_store.client import FeatureStoreClient  # noqa: E402
from src.feature_store.observability import (  # noqa: E402
    load_profiles,
    profile_snapshot,
    profile_stored_snapshot,
    trend,
)
from src.features.contract import MODEL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from src.features.pit_extractor import MultiDomainFeatureBuilder  # noqa: E402
from src.monitoring.metrics import refresh_feature_store_gauges  # noqa: E402
from src.monitoring.state import MonitoringState  # noqa: E402


@pytest.fixture
def builder(synthetic_data):
    return MultiDomainFeatureBuilder(synthetic_data)


def test_profile_vs_previous_snapshot(builder, tmp_path):
    summary = profile_snapshot(builder.features_at(180), 180, reference=builder.features_at(173),
                               reference_day=173, report_dir=tmp_path)
    assert summary["cutoff_day"] == 180 and summary["reference_day"] == 173
    assert set(summary["drift"]["columns"]) == set(MODEL_FEATURES)
    assert set(summary["out_of_range"]) == set(NUMERIC_FEATURES)
    assert {"mean", "std", "min", "max", "p25", "p50", "p75"} <= set(summary["numeric_stats"]["spend_90d"])
    assert (tmp_path / "snapshot_180.html").stat().st_size > 10_000
    assert load_profiles(tmp_path)[0]["cutoff_day"] == 180


def test_first_snapshot_has_statistics_only(builder, tmp_path):
    summary = profile_snapshot(builder.features_at(180), 180, report_dir=tmp_path, save_html=False)
    assert summary["drift"] is None and summary["out_of_range"] is None
    assert summary["numeric_stats"]["spend_90d"]["mean"] > 0
    assert not list(tmp_path.glob("*.html"))


def test_injected_shift_is_detected(builder, tmp_path):
    previous = builder.features_at(173)
    shifted = builder.features_at(180)
    shifted[["spend_30d", "spend_90d", "spend_180d"]] *= 100  # e.g. cents instead of dollars upstream
    summary = profile_snapshot(shifted, 180, reference=previous, reference_day=173,
                               report_dir=tmp_path, save_html=False)
    assert {"spend_30d", "spend_90d", "spend_180d"} <= set(summary["drift"]["drifted_columns"])
    assert summary["out_of_range"]["spend_180d"]["share"] > 0.5


def test_stored_snapshots_and_trend(builder, tmp_path):
    client = FeatureStoreClient(root=tmp_path / "fs")
    client.backfill(builder, [160, 167, 173, 180])
    for day in client.snapshot_days():
        profile_stored_snapshot(client, day, report_dir=tmp_path / "rep", save_html=False)
    table = trend("spend_90d", report_dir=tmp_path / "rep")
    assert table["cutoff_day"].tolist() == [160, 167, 173, 180]
    assert table["drift_score"].isna().tolist() == [True, False, False, False]  # first has no reference
    assert trend("classification_2", report_dir=tmp_path / "rep").shape[0] == 4


def _gauge(name, **labels):
    return REGISTRY.get_sample_value(name, labels or None)


def test_batch_publishes_quality_and_profile(builder, synthetic_raw_dir, synthetic_data, tmp_path):
    from src.pipelines.batch_pipeline import run_batch_pipeline
    from src.serving.batch_store import BatchScoreStore

    FeatureStoreClient().write_snapshot(builder.features_at(173), 173)  # previous snapshot in the store
    state = MonitoringState(redis_client=None, report_dir=tmp_path / "state")
    summary = run_batch_pipeline(
        cutoff_day=180, raw_data_dir=synthetic_raw_dir, data=synthetic_data, output_dir=tmp_path / "out",
        lakehouse_path=tmp_path / "lh" / "wh.duckdb", lapse_model_path=tmp_path / "none.joblib",
        batch_store=BatchScoreStore(redis_client=None), monitoring_state=state,
    )
    profile = summary["feature_store"]["profile"]
    assert profile["reference_day"] == 173
    assert (tmp_path / "out" / "reports" / "feature_store" / "snapshot_180.html").exists()
    assert state.latest("feature_store_quality") == {"cutoff_day": 180, "passed": True, "violations": []}
    assert state.latest("feature_store_profile")["cutoff_day"] == 180

    refresh_feature_store_gauges(state)
    assert _gauge("propensity_feature_store_quality_passed") == 1
    assert _gauge("propensity_feature_store_snapshot_day") == 180
    assert _gauge("propensity_feature_store_drift_share") == pytest.approx(profile["drifted_share"])
    assert _gauge("propensity_feature_store_out_of_range_share", feature_name="spend_90d") is not None


def test_batch_publishes_rejected_snapshot(synthetic_raw_dir, synthetic_data, tmp_path, monkeypatch):
    from src.features.quality import FeatureQualityError
    from src.pipelines.batch_pipeline import run_batch_pipeline
    from src.serving.batch_store import BatchScoreStore

    original = MultiDomainFeatureBuilder.features_at
    monkeypatch.setattr(MultiDomainFeatureBuilder, "features_at",
                        lambda self, cutoff: original(self, cutoff).assign(store_loyalty_index=2.0))
    state = MonitoringState(redis_client=None, report_dir=tmp_path / "state")
    with pytest.raises(FeatureQualityError):
        run_batch_pipeline(
            cutoff_day=180, raw_data_dir=synthetic_raw_dir, data=synthetic_data, output_dir=tmp_path / "out",
            lakehouse_path=tmp_path / "lh" / "wh.duckdb", lapse_model_path=tmp_path / "none.joblib",
            batch_store=BatchScoreStore(redis_client=None), monitoring_state=state,
        )
    quality = state.latest("feature_store_quality")
    assert quality["passed"] is False and quality["violations"]
    refresh_feature_store_gauges(state)
    assert _gauge("propensity_feature_store_quality_passed") == 0
    assert _gauge("propensity_feature_store_quality_violations") == len(quality["violations"])


def test_profiling_failure_does_not_stop_the_batch(synthetic_raw_dir, synthetic_data, tmp_path, monkeypatch):
    import src.feature_store.observability as obs
    from src.pipelines.batch_pipeline import run_batch_pipeline
    from src.serving.batch_store import BatchScoreStore

    def boom(*args, **kwargs):
        raise RuntimeError("evidently broke")

    monkeypatch.setattr(obs, "profile_stored_snapshot", boom)
    summary = run_batch_pipeline(
        cutoff_day=180, raw_data_dir=synthetic_raw_dir, data=synthetic_data, output_dir=tmp_path / "out",
        lakehouse_path=tmp_path / "lh" / "wh.duckdb", lapse_model_path=tmp_path / "none.joblib",
        batch_store=BatchScoreStore(redis_client=None),
        monitoring_state=MonitoringState(redis_client=None, report_dir=tmp_path / "state"),
    )
    assert summary["total_households_scored"] == len(synthetic_data.households)
    assert "profile" not in summary["feature_store"]


def test_profiling_can_be_disabled(synthetic_raw_dir, synthetic_data, tmp_path, monkeypatch):
    from src.pipelines.batch_pipeline import run_batch_pipeline
    from src.serving.batch_store import BatchScoreStore

    monkeypatch.setattr(settings, "FEATURE_STORE_PROFILING", False)
    summary = run_batch_pipeline(
        cutoff_day=180, raw_data_dir=synthetic_raw_dir, data=synthetic_data, output_dir=tmp_path / "out",
        lakehouse_path=tmp_path / "lh" / "wh.duckdb", lapse_model_path=tmp_path / "none.joblib",
        batch_store=BatchScoreStore(redis_client=None),
        monitoring_state=MonitoringState(redis_client=None, report_dir=tmp_path / "state"),
    )
    assert "profile" not in summary["feature_store"]
    assert not (tmp_path / "out" / "reports" / "feature_store").exists()
