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

    # Master Dataset (MinIO / S3)
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_MASTER: str = "lakehouse-master"
    S3_REGION: str = "us-east-1"
    S3_SECURE: bool = False

    # OLTP Database (PostgreSQL)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "propensity_oltp"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres_secure_pwd"

    # Speed & Serving Layer (Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_TTL_SECONDS: int = 86400  # 24 Hours


settings = Settings()
