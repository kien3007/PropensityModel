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

    # Feature store (Feast). Every batch run appends its snapshot to the offline history and
    # training reads features back with point-in-time joins.
    FEATURE_STORE_ENABLED: bool = True
    FEATURE_STORE_DIR: Path = BASE_DIR / "data/feature_store"
    FEATURE_TTL_DAYS: int = 7                  # snapshot cadence: older features are treated as missing
    TRAINING_FEATURE_SOURCE: str = "feast"     # "feast" (historical store) or "builder" (recompute)
    FEATURE_QUALITY_CHECKS: bool = True        # quality gate (src/features/quality.py) before store/score
    FEATURE_STORE_PROFILING: bool = True       # Evidently report of each new snapshot vs the previous one

    # Production hardening. STRICT_INFRA=true disables every silent in-memory fallback:
    # the service fails fast if Redis is unreachable.
    STRICT_INFRA: bool = False

    # Serving API security. Comma-separated keys accepted in the X-API-Key header for write,
    # admin and bulk-export endpoints. Empty = open in development, rejected in production.
    SERVING_API_KEYS: str = ""
    RATE_LIMIT_PER_MINUTE: int = 600

    # Canary: optional challenger bundle scored by the batch layer alongside the champion.
    CHALLENGER_MODEL_PATH: Optional[Path] = None

    # Monitoring & automated retraining triggers.
    DRIFT_PSI_ALERT: float = 0.20          # Prometheus early-warning threshold per feature
    REALIZED_AUC_DROP_ALERT: float = 0.05  # expected (test) AUC - realized AUC that counts as degraded
    AUTO_RETRAIN_ON_DEGRADATION: bool = True



settings = Settings()
