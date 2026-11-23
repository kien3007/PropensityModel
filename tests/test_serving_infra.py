"""
Serving infrastructure: Redis-backed batch store (atomic snapshot swap), bootstrap from batch
artefacts, idempotent speed layer on Redis, STRICT_INFRA fail-fast, canary contract checks and
challenger scoring in the batch pipeline.
"""
import json

import fakeredis
import pandas as pd
import pytest

from src.common import redis_client
from src.common.redis_client import InfrastructureUnavailable, connect_redis
from src.config.settings import settings
from src.models.predictor import PropensityPredictor
from src.serving.batch_store import BatchScoreStore
from src.serving.canary_router import CanaryRouter
from src.speed_layer import delta_stream
from src.speed_layer.delta_stream import SpeedLayerProcessor
from src.speed_layer.kafka_bus import KafkaEventProducer


def _scores(n=5, p=0.5):
    return pd.DataFrame({
        "household_key": range(1, n + 1),
        "purchase_probability": [p] * n,
        "rank": range(1, n + 1),
        "decile": [10] * n,
        "urgency_tier": ["REGULAR"] * n,
        "marketing_action": ["STANDARD_ENGAGEMENT"] * n,
    })


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


def test_redis_store_swaps_snapshots_atomically(fake_redis):
    store = BatchScoreStore(redis_client=fake_redis)
    store.publish(_scores(5, 0.2), "champion", {"run_id": "r1", "snapshot_day": 700})
    store.publish(_scores(3, 0.9), "champion", {"run_id": "r2", "snapshot_day": 711})

    assert store.backend == "redis"
    assert store.count("champion") == 3  # old snapshot fully replaced, not merged
    assert store.get(1)["batch_probability"] == pytest.approx(0.9)
    assert store.get(5) is None
    assert store.metadata()["run_id"] == "r2"
    assert not [k for k in fake_redis.keys("*") if "staging" in k]

    # A second store instance (another serving replica) sees the same data.
    replica = BatchScoreStore(redis_client=fake_redis)
    assert replica.get(2)["batch_probability"] == pytest.approx(0.9)
    assert [r["household_key"] for r in replica.records(limit=2)] == [1, 2]


def test_store_rejects_incomplete_scores():
    with pytest.raises(ValueError, match="missing columns"):
        BatchScoreStore(redis_client=None).publish(_scores().drop(columns=["decile"]))


def test_bootstrap_from_latest_batch_manifest(tmp_path):
    champion = tmp_path / "scores.parquet"
    challenger = tmp_path / "challenger.parquet"
    _scores(4, 0.4).to_parquet(champion)
    _scores(4, 0.6).to_parquet(challenger)
    (tmp_path / "batch_manifest_day_711.json").write_text(json.dumps({
        "run_id": "B1", "cutoff_day": 711, "model_bundle": "champ", "challenger_bundle": "chal",
        "artifacts": {"parquet": str(champion), "challenger_parquet": str(challenger)},
    }), encoding="utf-8")

    store = BatchScoreStore(redis_client=None)
    assert store.bootstrap_from_outputs(tmp_path) is True
    assert store.count("champion") == 4 and store.count("challenger") == 4
    assert store.metadata("champion")["snapshot_day"] == 711
    assert store.bootstrap_from_outputs(tmp_path) is False  # never overwrites a populated store


def test_speed_layer_on_redis_is_idempotent(fake_redis, monkeypatch):
    monkeypatch.setattr(delta_stream, "connect_redis", lambda component: fake_redis)
    speed = SpeedLayerProcessor()
    assert speed.backend == "redis"
    event = {"household_key": 9, "event_type": "coupon_clip", "event_id": "e-1"}
    speed.process_event(event)
    again = speed.process_event(event)
    assert again["duplicate"] is True and again["intraday_coupons_clipped"] == 1
    assert speed.active_delta_count() == 1
    speed.reset_intraday_deltas()
    assert speed.active_delta_count() == 0
    assert fake_redis.exists("realtime:seen:e-1")  # dedup marker survives the delta reset


def test_strict_infra_fails_fast(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_INFRA", True)
    monkeypatch.setattr(settings, "REDIS_PORT", 1)  # nothing listens there
    with pytest.raises(InfrastructureUnavailable, match="Redis"):
        connect_redis("test")
    with pytest.raises(InfrastructureUnavailable, match="Kafka"):
        KafkaEventProducer(bootstrap_servers="127.0.0.1:1", enabled=True, fallback_processor=object())


def test_non_strict_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_INFRA", False)
    monkeypatch.setattr(settings, "REDIS_PORT", 1)
    assert redis_client.connect_redis("test") is None


def test_canary_rejects_challenger_with_different_target():
    champion = PropensityPredictor()
    router = CanaryRouter(champion_predictor=champion)
    lapse_path = settings.BASE_DIR / "models" / "lapse" / "champion_model.joblib"
    assert router.load_challenger(lapse_path) is False
    assert router.challenger is None and router.enabled is False
    assert router.load_challenger(settings.MODEL_PATH) is True


def test_batch_pipeline_publishes_champion_and_challenger(synthetic_raw_dir, synthetic_data, tmp_path):
    from src.pipelines.batch_pipeline import run_batch_pipeline

    store = BatchScoreStore(redis_client=None)
    summary = run_batch_pipeline(
        cutoff_day=180,
        raw_data_dir=synthetic_raw_dir,
        data=synthetic_data,
        output_dir=tmp_path / "out",
        lakehouse_path=tmp_path / "lh" / "wh.duckdb",
        lapse_model_path=tmp_path / "no_lapse.joblib",
        challenger_model_path=settings.MODEL_PATH,
        batch_store=store,
    )
    n = len(synthetic_data.households)
    assert store.count("champion") == n and store.count("challenger") == n
    assert store.metadata("champion")["snapshot_day"] == 180
    assert summary["challenger"]["model_bundle"] == summary["model_bundle"]
    assert "challenger_parquet" in summary["artifacts"]
