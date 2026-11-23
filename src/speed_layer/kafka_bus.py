"""
Kafka Streaming Message Bus Module for High-Throughput POS & App Events.
Implements dual-dispatch streaming producer with graceful Redis fallback,
and scalable background consumer worker.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import uuid
from typing import Any, Dict, Optional

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_LIB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KAFKA_LIB_AVAILABLE = False

from src.common.redis_client import InfrastructureUnavailable
from src.config.settings import settings
from src.speed_layer.delta_stream import SpeedLayerProcessor

logger = logging.getLogger("KafkaEventBus")


def _is_broker_reachable(bootstrap_servers: str, timeout: float = 0.2) -> bool:
    """Fast pre-flight check to verify if Kafka broker port is listening without blocking."""
    try:
        first_broker = bootstrap_servers.split(",")[0].strip()
        parts = first_broker.split(":")
        host = parts[0]
        if host in ("localhost", "127.0.0.1", "0.0.0.0"):
            host = "127.0.0.1"
        port = int(parts[1]) if len(parts) > 1 else 9092
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            return True
    except Exception:
        return False


class KafkaEventProducer:
    """
    High-throughput event producer publishing streaming POS & App events to Kafka.
    Falls back gracefully to direct Redis ingestion if Kafka cluster is unavailable.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        enabled: Optional[bool] = None,
        fallback_processor: Optional[SpeedLayerProcessor] = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self.topic = topic or settings.KAFKA_TOPIC_EVENTS
        self.enabled = enabled if enabled is not None else settings.KAFKA_ENABLED
        self._producer: Optional[Any] = None
        self._fallback_processor: Optional[SpeedLayerProcessor] = fallback_processor

        if self.enabled and KAFKA_LIB_AVAILABLE:
            self._init_producer()

    def _init_producer(self) -> None:
        if not _is_broker_reachable(self.bootstrap_servers, timeout=0.5):
            if settings.STRICT_INFRA:
                raise InfrastructureUnavailable(f"Kafka broker not reachable at {self.bootstrap_servers}")
            logger.info(
                f"Kafka broker not listening at {self.bootstrap_servers}. "
                "Activating direct Redis SpeedLayer fallback."
            )
            self._producer = None
            return

        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
                acks="all",
                retries=3,
                request_timeout_ms=1000,
                api_version_auto_timeout_ms=1000,
            )
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers} (Topic: {self.topic})")
        except Exception as e:
            if settings.STRICT_INFRA:
                raise InfrastructureUnavailable(f"Kafka producer init failed: {e}") from e
            logger.warning(
                f"Kafka broker not accessible at {self.bootstrap_servers} ({e}). "
                "Activating direct Redis SpeedLayer fallback."
            )
            self._producer = None

    def publish_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publishes event to Kafka topic or processes immediately via Redis fallback.
        Every event gets an `event_id` (kept if the client sent one) so that redelivery and
        client retries are applied at most once by the speed layer.
        """
        event_data = {**event_data, "event_id": event_data.get("event_id") or uuid.uuid4().hex}
        hh_key = event_data.get("household_key")

        # 1. Try Kafka publish
        if self._producer is not None:
            try:
                future = self._producer.send(self.topic, key=hh_key, value=event_data)
                record_metadata = future.get(timeout=2.0)
                logger.debug(
                    f"Published event to Kafka topic {record_metadata.topic} "
                    f"partition {record_metadata.partition} offset {record_metadata.offset}"
                )
                return {
                    "event_id": event_data["event_id"],
                    "dispatched_to": "kafka",
                    "topic": self.topic,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset,
                    "status": "QUEUED",
                }
            except Exception as e:
                if settings.STRICT_INFRA:
                    raise InfrastructureUnavailable(f"Kafka publish failed: {e}") from e
                logger.warning(f"Failed to publish to Kafka ({e}), falling back to direct Redis.")

        # 2. Fallback: Direct Redis Ingestion
        if self._fallback_processor is None:
            self._fallback_processor = SpeedLayerProcessor()

        delta = self._fallback_processor.process_event(event_data)
        return {
            "event_id": event_data["event_id"],
            "dispatched_to": "direct_redis_fallback",
            "status": "ACCEPTED",
            "updated_delta": delta,
        }

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()


class KafkaEventConsumerWorker:
    """
    Scalable background worker consuming events from Kafka topic and updating Redis Speed Layer.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self.topic = topic or settings.KAFKA_TOPIC_EVENTS
        self.group_id = group_id or settings.KAFKA_CONSUMER_GROUP
        self.processor = SpeedLayerProcessor()
        self._running = False

    def start_consuming(self, max_messages: Optional[int] = None) -> int:
        """
        Starts the event polling loop. Returns number of messages processed.
        """
        if not KAFKA_LIB_AVAILABLE:
            raise RuntimeError("kafka-python library not available.")

        logger.info(f"Connecting Consumer to {self.bootstrap_servers} (Topic: {self.topic}, Group: {self.group_id})...")
        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers.split(","),
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=5000,
        )

        self._running = True
        processed = 0

        def shutdown(signum, frame):
            logger.info("Shutdown signal received for Kafka consumer.")
            self._running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            logger.info("Kafka Consumer is listening for real-time POS & App events...")
            while self._running:
                records = consumer.poll(timeout_ms=1000)
                for topic_partition, messages in records.items():
                    for msg in messages:
                        event = msg.value
                        delta = self.processor.process_event(event)
                        processed += 1
                        logger.debug(f"Consumed event from Kafka for HH {event.get('household_key')}: {delta}")

                        if max_messages and processed >= max_messages:
                            self._running = False
                            break
        finally:
            consumer.close()
            logger.info(f"Kafka consumer closed. Total processed: {processed} events.")

        return processed


def main():
    parser = argparse.ArgumentParser(description="Propensity Kafka Stream Bus Runner")
    parser.add_argument("--mode", choices=["consumer", "test-produce"], default="consumer")
    parser.add_argument("--count", type=int, default=10, help="Number of test messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.mode == "test-produce":
        producer = KafkaEventProducer()
        for i in range(args.count):
            res = producer.publish_event({
                "household_key": (i % 2500) + 1,
                "event_type": "coupon_clip",
                "coupon_id": 1000 + i,
            })
            print(f"Produced [{i+1}/{args.count}]: {res}")
        producer.close()
    elif args.mode == "consumer":
        worker = KafkaEventConsumerWorker()
        worker.start_consuming()


if __name__ == "__main__":
    main()
