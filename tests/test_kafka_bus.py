"""
Unit tests for Kafka Event Bus and Streaming Speed Layer Dispatch.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.speed_layer.kafka_bus import KafkaEventProducer


def test_kafka_producer_direct_redis_fallback():
    # Producer without broker connection should seamlessly fall back to Redis
    producer = KafkaEventProducer(bootstrap_servers="127.0.0.1:9999", enabled=True)

    event = {
        "household_key": 100,
        "event_type": "coupon_clip",
        "coupon_id": 555,
    }

    result = producer.publish_event(event)
    assert result["status"] == "ACCEPTED"
    assert result["dispatched_to"] == "direct_redis_fallback"
    assert "updated_delta" in result
    assert result["updated_delta"]["intraday_coupons_clipped"] >= 1


def test_kafka_producer_mocked_publish():
    producer = KafkaEventProducer(enabled=False)

    # Mock the internal KafkaProducer
    mock_future = MagicMock()
    mock_meta = MagicMock()
    mock_meta.topic = "pos-intraday-events"
    mock_meta.partition = 0
    mock_meta.offset = 42
    mock_future.get.return_value = mock_meta

    mock_kafka = MagicMock()
    mock_kafka.send.return_value = mock_future
    producer._producer = mock_kafka

    event = {
        "household_key": 200,
        "event_type": "pos_checkout",
        "sales_value": 85.50,
    }

    result = producer.publish_event(event)
    assert result["dispatched_to"] == "kafka"
    assert result["status"] == "QUEUED"
    assert result["offset"] == 42
    assert mock_kafka.send.called
