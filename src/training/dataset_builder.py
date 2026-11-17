"""
Dataset Builder Module for Model Training and Retraining.
Constructs temporal multi-snapshot training and validation datasets with prospective binary labels.
"""
from __future__ import annotations

import logging
from typing import List, Optional
import pandas as pd

from src.features.pit_extractor import PointInTimeFeatureExtractor, MODEL_FEATURES

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds time-split datasets across multiple historical cutoffs without future leakage.
    """

    @staticmethod
    def build_snapshot_with_target(
        transactions: pd.DataFrame,
        cutoff_day: int,
        horizon_days: int = 7,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Extracts features strictly for DAY <= cutoff_day, and sets target Y in (cutoff, cutoff + horizon_days].
        Target Y = 1 if household made at least 1 purchase in the subsequent horizon; else 0.
        """
        feats = feature_cols or MODEL_FEATURES
        snapshot = PointInTimeFeatureExtractor.extract_features(
            transactions=transactions,
            cutoff=cutoff_day,
            feature_cols=feats,
        )

        # Labeling: Active buyers in prospective horizon (cutoff, cutoff + horizon_days]
        future_tx = transactions.loc[
            (transactions["DAY"] > cutoff_day) & (transactions["DAY"] <= cutoff_day + horizon_days)
        ]
        future_buyers = set(future_tx["household_key"].unique())

        snapshot["target"] = snapshot["household_key"].isin(future_buyers).astype(int)
        logger.info(
            f"Built snapshot DAY={cutoff_day} (Horizon={horizon_days}d): "
            f"{len(snapshot)} households, positive rate: {snapshot['target'].mean():.4f}"
        )
        return snapshot

    @staticmethod
    def build_multi_cutoff_dataset(
        transactions: pd.DataFrame,
        cutoffs: List[int],
        horizon_days: int = 7,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generates and stacks snapshots across multiple cutoff days.
        """
        frames = []
        for cutoff in sorted(cutoffs):
            df = DatasetBuilder.build_snapshot_with_target(
                transactions=transactions,
                cutoff_day=cutoff,
                horizon_days=horizon_days,
                feature_cols=feature_cols,
            )
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            f"Built multi-cutoff dataset: {len(combined)} rows across cutoffs {cutoffs}, "
            f"overall positive rate: {combined['target'].mean():.4f}"
        )
        return combined
