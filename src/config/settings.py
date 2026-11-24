"""
Central configuration settings module for Propensity Enterprise ML System.
Uses Pydantic Settings for environment validation and type safety.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    # App
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_PORT: int = 8000
    APP_HOST: str = "0.0.0.0"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "sample_data/dunnhumby"
    LAKEHOUSE_PATH: Path = BASE_DIR / "data/lakehouse/propensity_warehouse.duckdb"
    MODEL_PATH: Path = BASE_DIR / "models/champion_model.joblib"
    OUTPUT_DIR: Path = BASE_DIR / "data/outputs"

    # Where training and batch scoring read raw data from. "lakehouse" = Bronze Parquet
    # (ingested from DATA_RAW_DIR on demand), "csv" = DATA_RAW_DIR directly.
    DATA_SOURCE: str = "lakehouse"

    # Speed & Serving Layer (Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_TTL_SECONDS: int = 86400  # 24 Hours

    # Feature store (Feast). Every batch run appends its snapshot to the offline history and
    # training reads features back with point-in-time joins.
    FEATURE_STORE_ENABLED: bool = True
    FEATURE_STORE_DIR: Path = BASE_DIR / "data/feature_store"
    FEATURE_TTL_DAYS: int = 7                  # snapshot cadence: older features are treated as missing
    TRAINING_FEATURE_SOURCE: str = "feast"     # "feast" (historical store) or "builder" (recompute)
    FEATURE_QUALITY_CHECKS: bool = True        # quality gate (src/features/quality.py) before store/score
    FEATURE_STORE_PROFILING: bool = True       # Evidently report of each new snapshot vs the previous one

    # Production hardening. STRICT_INFRA=true disables every silent in-memory fallback:
    # the service fails fast if Redis (or an enabled Kafka) is unreachable.
    STRICT_INFRA: bool = False
    EVENT_DEDUP_TTL_SECONDS: int = 86400

    # Serving API security. Comma-separated keys accepted in the X-API-Key header for write,
    # admin and bulk-export endpoints. Empty = open in development, rejected in production.
    SERVING_API_KEYS: str = ""
    RATE_LIMIT_PER_MINUTE: int = 600

    # Canary: optional challenger bundle scored by the batch layer alongside the champion.
    CHALLENGER_MODEL_PATH: Optional[Path] = None

    # Lambda fusion rules (HEURISTIC, not learned; to be calibrated from feedback data).
    FUSION_PURCHASE_DAMPENING: float = 0.25    # multiply P after an intraday checkout
    FUSION_COUPON_BOOST_PER_CLIP: float = 0.15  # add per coupon clipped today
    FUSION_APP_NUDGE: float = 0.05             # add after repeated app interactions
    FUSION_APP_MIN_INTERACTIONS: int = 2

    # Monitoring & automated retraining triggers.
    DRIFT_PSI_ALERT: float = 0.20          # Prometheus early-warning threshold per feature
    REALIZED_AUC_DROP_ALERT: float = 0.05  # expected (test) AUC - realized AUC that counts as degraded
    AUTO_RETRAIN_ON_DEGRADATION: bool = True
    PREDICTION_LOG_MAXLEN: int = 100_000   # served predictions kept for fusion-rule evaluation
    CONSUMER_METRICS_PORT: int = 9101      # Prometheus exporter of the Kafka consumer worker

    # Streaming Event Bus (Kafka)
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_EVENTS: str = "pos-intraday-events"
    KAFKA_CONSUMER_GROUP: str = "propensity-speed-consumer-group"
    KAFKA_ENABLED: bool = True


settings = Settings()
