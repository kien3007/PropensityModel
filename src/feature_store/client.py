"""
Feast-backed historical feature store.

Layout under settings.FEATURE_STORE_DIR:
  offline/household_features.parquet   every snapshot ever written (one per cutoff day)
  registry.db                          Feast registry (applied from code; no `feast apply` needed)

Writes:  `write_snapshot` runs the feature quality gate, then appends (or replaces) the rows of one cutoff atomically; it never
         drops other snapshots, so the offline store accumulates real history.
Reads:   `get_historical_features` is a Feast point-in-time join (entity rows at a cutoff get
         the latest snapshot with timestamp <= cutoff, within the TTL).
There is no online store: serving uses precomputed batch scores (Redis), not live features.
`FeatureStoreSource` gives training the same interface as MultiDomainFeatureBuilder, reading
features from the store and backfilling cutoffs that were never snapshotted.
"""
from __future__ import annotations

import datetime
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from feast import FeatureStore
from feast.repo_config import RepoConfig

from src.config.settings import settings
from src.feature_store.definitions import (
    FEATURE_VIEW_NAME,
    PROJECT_NAME,
    build_feature_view,
    day_to_timestamp,
    household,
)
from src.features.contract import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    FeatureContractError,
    normalize_feature_dtypes,
    validate_feature_frame,
)
from src.features.quality import check_feature_quality

logger = logging.getLogger("FeatureStore")

OFFLINE_COLUMNS = ["household_key", "cutoff_day", "event_timestamp", "created_timestamp", *MODEL_FEATURES]


class MissingSnapshotError(LookupError):
    """Raised when features are requested for a cutoff that was never snapshotted."""


class FeatureStoreClient:
    def __init__(self, root: Optional[Path | str] = None, ttl_days: Optional[int] = None) -> None:
        self.root = Path(root or settings.FEATURE_STORE_DIR).resolve()
        self.offline_path = self.root / "offline" / "household_features.parquet"
        self.offline_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days or settings.FEATURE_TTL_DAYS
        config = RepoConfig(
            project=PROJECT_NAME,
            provider="local",
            registry=str(self.root / "registry.db"),
            offline_store={"type": "file"},
            online_store=None,  # no online store: nothing is materialized or served live
            entity_key_serialization_version=3,
            repo_path=self.root,
        )
        self.store = FeatureStore(config=config)
        self.feature_view = build_feature_view(self.offline_path.as_posix(), self.ttl_days)
        self._applied = False

    # -------------------------------------------------------------- registry
    def apply(self) -> None:
        """Registers entity + feature view (idempotent). Needs the offline file to exist."""
        if not self._applied:
            self.store.apply([household, self.feature_view])
            self._applied = True

    # ----------------------------------------------------------------- write
    def _read_offline(self) -> pd.DataFrame:
        if not self.offline_path.exists():
            return pd.DataFrame(columns=OFFLINE_COLUMNS)
        return pd.read_parquet(self.offline_path)

    def snapshot_days(self) -> List[int]:
        if not self.offline_path.exists():
            return []
        days = pd.read_parquet(self.offline_path, columns=["cutoff_day"])["cutoff_day"]
        return sorted(int(d) for d in days.unique())

    def previous_snapshot_day(self, cutoff_day: int) -> Optional[int]:
        earlier = [d for d in self.snapshot_days() if d < int(cutoff_day)]
        return earlier[-1] if earlier else None

    def read_snapshot(self, cutoff_day: int) -> pd.DataFrame:
        """The stored rows of exactly one cutoff (household_key + contract features)."""
        if not self.offline_path.exists():
            raise MissingSnapshotError("Feature store is empty")
        df = pd.read_parquet(self.offline_path, filters=[("cutoff_day", "==", int(cutoff_day))])
        if df.empty:
            raise MissingSnapshotError(f"No feature snapshot for DAY {cutoff_day}")
        return df[["household_key", *MODEL_FEATURES]].reset_index(drop=True)

    @staticmethod
    def _previous_snapshot(existing: pd.DataFrame, cutoff_day: int) -> Optional[pd.DataFrame]:
        """Rows of the latest stored snapshot strictly before `cutoff_day` (None if there is none)."""
        if not len(existing):
            return None
        earlier = existing.loc[existing["cutoff_day"] < int(cutoff_day), "cutoff_day"]
        if earlier.empty:
            return None
        return existing[existing["cutoff_day"] == earlier.max()]

    def write_snapshot(self, features: pd.DataFrame, cutoff_day: int) -> int:
        """
        Appends one cutoff's snapshot to the offline history (replacing that cutoff if present).
        The snapshot must pass the quality gate first (compared with the previous stored
        snapshot); a failing snapshot raises FeatureQualityError and nothing is written.
        """
        validate_feature_frame(features)
        snap = normalize_feature_dtypes(features)[["household_key", *MODEL_FEATURES]].copy()
        for col in CATEGORICAL_FEATURES:  # Feast String columns are stored as plain strings
            snap[col] = snap[col].astype(str)

        existing = self._read_offline()
        if settings.FEATURE_QUALITY_CHECKS:
            check_feature_quality(snap, self._previous_snapshot(existing, cutoff_day), cutoff_day=int(cutoff_day))

        snap.insert(1, "cutoff_day", int(cutoff_day))
        snap.insert(2, "event_timestamp", day_to_timestamp(cutoff_day))
        snap.insert(3, "created_timestamp", datetime.datetime.now(datetime.timezone.utc))
        kept = existing[existing["cutoff_day"] != int(cutoff_day)] if len(existing) else existing
        combined = pd.concat([kept, snap], ignore_index=True) if len(kept) else snap
        combined = combined.sort_values(["cutoff_day", "household_key"]).reset_index(drop=True)
        fd, tmp = tempfile.mkstemp(dir=self.offline_path.parent, suffix=".parquet.tmp")
        os.close(fd)
        try:
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, self.offline_path)
        finally:
            Path(tmp).unlink(missing_ok=True)

        self.apply()
        logger.info(f"Feature snapshot DAY={cutoff_day}: {len(snap):,} households "
                    f"(history: {combined['cutoff_day'].nunique()} snapshots).")
        return len(snap)

    def backfill(self, builder, cutoffs: Iterable[int], overwrite: bool = False) -> List[int]:
        """Computes and stores snapshots for cutoffs that are missing (or all, with overwrite)."""
        existing = set(self.snapshot_days())
        written = []
        for cutoff in sorted(set(int(c) for c in cutoffs)):
            if overwrite or cutoff not in existing:
                self.write_snapshot(builder.features_at(cutoff), cutoff)
                written.append(cutoff)
        return written

    # ------------------------------------------------------------------ read
    def get_historical_features(self, households: Sequence[int], cutoff_day: int) -> pd.DataFrame:
        """Point-in-time join through Feast for every household at one cutoff."""
        if not self.offline_path.exists():
            raise MissingSnapshotError("Feature store is empty; run a batch or backfill first")
        self.apply()
        entity_df = pd.DataFrame({"household_key": list(households)})
        entity_df["event_timestamp"] = day_to_timestamp(cutoff_day)
        refs = [f"{FEATURE_VIEW_NAME}:{f}" for f in MODEL_FEATURES]
        df = self.store.get_historical_features(entity_df=entity_df, features=refs).to_df()
        df = df.drop(columns=["event_timestamp"]).assign(cutoff_day=int(cutoff_day))
        df["household_key"] = df["household_key"].astype("int64")
        order = pd.DataFrame({"household_key": [int(h) for h in households]})
        # Feast drops entity rows with no snapshot inside the TTL; the left join exposes them.
        df = order.merge(df, on="household_key", how="left", validate="one_to_one")
        missing = df[MODEL_FEATURES].isna().any(axis=1)
        if missing.any():
            raise MissingSnapshotError(
                f"{int(missing.sum())} of {len(df)} households have no snapshot within {self.ttl_days} days "
                f"before DAY {cutoff_day}; backfill that cutoff"
            )
        df = normalize_feature_dtypes(df[["household_key", "cutoff_day", *MODEL_FEATURES]])
        validate_feature_frame(df)
        return df


class FeatureStoreSource:
    """
    Drop-in replacement for MultiDomainFeatureBuilder in DatasetBuilder: features come from the
    feature store's point-in-time join; labels, baskets and the household universe still come
    from the builder (labels need data after the cutoff, which a feature store must not hold).
    """

    def __init__(self, builder, client: Optional[FeatureStoreClient] = None, backfill_missing: bool = True) -> None:
        self.builder = builder
        self.client = client or FeatureStoreClient()
        self.backfill_missing = backfill_missing
        self.source = "feast"

    @property
    def data(self):
        return self.builder.data

    @property
    def baskets(self) -> pd.DataFrame:
        return self.builder.baskets

    @property
    def households(self) -> List[int]:
        return self.builder.households

    def last_observed_day(self) -> int:
        return self.builder.last_observed_day()

    def labels_at(self, cutoff: int, horizon_days: int) -> pd.DataFrame:
        return self.builder.labels_at(cutoff, horizon_days)

    def features_at(self, cutoff: int) -> pd.DataFrame:
        if cutoff not in self.client.snapshot_days():
            if not self.backfill_missing:
                raise MissingSnapshotError(f"No feature snapshot for DAY {cutoff}")
            logger.info(f"Backfilling missing feature snapshot DAY={cutoff}")
            self.client.write_snapshot(self.builder.features_at(cutoff), cutoff)
        try:
            return self.client.get_historical_features(self.households, cutoff)
        except MissingSnapshotError as exc:
            raise FeatureContractError(str(exc)) from exc
