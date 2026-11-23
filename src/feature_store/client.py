"""
Feast Feature Store Client Integration Module.
Provides high-level programmatic access for:
- Point-in-Time Historical Feature Joins (Offline Store)
- Sub-millisecond Online Feature Retrieval (Online Store)
- Snapshot Materialization from Lakehouse to Online Store
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from feast import FeatureStore
from src.config.settings import settings
from src.data.loader import DataLoader
from src.features.contract import MODEL_FEATURES as MULTIDOMAIN_FEATURES_28
from src.features.pit_extractor import MultiDomainFeatureBuilder

logger = logging.getLogger("FeatureStoreClient")

FEATURE_REPO_PATH = Path(__file__).resolve().parent.parent.parent / "feature_repo"


class FeastFeatureStoreClient:
    """
    Enterprise Feast client managing Offline/Online stores and Lakehouse sync.
    """

    def __init__(self, repo_path: Optional[Path | str] = None) -> None:
        self.repo_path = Path(repo_path or FEATURE_REPO_PATH)
        self.store = FeatureStore(repo_path=str(self.repo_path))

    def get_feature_view_names(self) -> List[str]:
        return [fv.name for fv in self.store.list_feature_views()]

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Retrieves historical features for entity DataFrame using Point-in-Time join.
        entity_df must contain ['household_key', 'event_timestamp'].
        """
        if feature_refs is None:
            feature_refs = [f"household_features_view:{f}" for f in MULTIDOMAIN_FEATURES_28]

        training_data = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        )
        return training_data.to_df()

    def get_online_features(
        self,
        household_keys: List[int],
        feature_refs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves online features for specified households.
        """
        if feature_refs is None:
            feature_refs = [f"household_features_view:{f}" for f in MULTIDOMAIN_FEATURES_28]

        entity_rows = [{"household_key": k} for k in household_keys]
        response = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )
        return response.to_dict()

    def materialize_snapshot(
        self,
        cutoff_day: int = 711,
        raw_data_dir: Optional[Path | str] = None,
        gold_dir: Optional[Path | str] = None,
    ) -> Path:
        """
        Generates snapshot with event_timestamp, saves to Gold Lakehouse,
        and materializes into Feast Online Store.
        """
        gold_path = Path(gold_dir or (settings.BASE_DIR / "data" / "lakehouse" / "gold"))
        gold_path.mkdir(parents=True, exist_ok=True)
        parquet_file = gold_path / "fct_household_features_snapshot.parquet"

        logger.info(f"Extracting 28 multi-domain features for cutoff day {cutoff_day}...")
        data = DataLoader.from_csv(raw_data_dir) if raw_data_dir is not None else DataLoader.load()
        snapshot_df = MultiDomainFeatureBuilder(data).features_at(cutoff_day)

        # Feast requires event_timestamp column
        base_date = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        as_of_date = base_date + datetime.timedelta(days=cutoff_day)
        snapshot_df["event_timestamp"] = as_of_date
        snapshot_df["created_timestamp"] = datetime.datetime.now(datetime.timezone.utc)

        snapshot_df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved Feast Gold snapshot ({len(snapshot_df)} rows) to {parquet_file}")

        # Materialize into Feast online store
        start_date = as_of_date - datetime.timedelta(days=1)
        end_date = as_of_date + datetime.timedelta(days=1)
        logger.info(f"Materializing features from {start_date} to {end_date}...")
        self.store.materialize(start_date=start_date, end_date=end_date)
        logger.info("Feast materialization to Online Store completed successfully.")

        return parquet_file
