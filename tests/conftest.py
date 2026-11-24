"""
Shared pytest configuration.

- `requires_data` tests are skipped automatically when the real Dunnhumby CSVs are absent
  (e.g. in CI), instead of failing.
- Synthetic Dunnhumby-shaped fixtures let pipeline tests run in seconds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config.model_config import TrainingConfig
from src.config.settings import settings
from src.data.loader import DataLoader, RawDataBundle
from tests.fixtures.synthetic_dunnhumby import make_raw_tables, write_raw_csvs

REAL_DATA_AVAILABLE = (Path(settings.DATA_RAW_DIR) / "transaction_data.csv").exists()


def pytest_collection_modifyitems(config, items):
    if REAL_DATA_AVAILABLE:
        return
    skip = pytest.mark.skip(reason=f"Real Dunnhumby data not found under {settings.DATA_RAW_DIR}")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_feature_store(tmp_path, monkeypatch):
    """Every test gets its own empty feature store so snapshots never leak between datasets."""
    monkeypatch.setattr(settings, "FEATURE_STORE_DIR", tmp_path / "feature_store")


@pytest.fixture(scope="session")
def synthetic_tables():
    return make_raw_tables()


@pytest.fixture(scope="session")
def synthetic_raw_dir(tmp_path_factory, synthetic_tables) -> Path:
    return write_raw_csvs(synthetic_tables, tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def synthetic_data(synthetic_raw_dir) -> RawDataBundle:
    return DataLoader.from_csv(synthetic_raw_dir)


@pytest.fixture
def small_training_config() -> TrainingConfig:
    """Temporal splits that fit the 200-day synthetic history, with a fast booster."""
    return TrainingConfig(
        horizon_days=7,
        lookback_days=60,
        fit_cutoffs=(70, 80, 90, 100, 110, 120),
        val_cutoffs=(135, 145),
        test_cutoffs=(165, 175),
        lgbm_params={
            "objective": "binary", "n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7,
            "min_child_samples": 5, "random_state": 42, "n_jobs": 1, "verbosity": -1,
        },
        early_stopping_rounds=10,
        calibration_cv_folds=3,
    )
