"""
Drift reference for the active champion.

The reference is the champion's own out-of-time test window (bundle `test_cutoffs`): the
most recent data the model was validated on. Features are recomputed with the shared
feature builder (eligible households of the champion's target) and scored by the champion,
so feature and prediction drift are measured against what the model was accepted on, not
against the current snapshot itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.features.targets import get_target
from src.models.predictor import PropensityPredictor


@dataclass
class DriftReference:
    features: pd.DataFrame
    scores: np.ndarray
    cutoffs: List[int]


def build_champion_reference(predictor: PropensityPredictor, builder: MultiDomainFeatureBuilder) -> DriftReference:
    cutoffs = [int(c) for c in (predictor.bundle.get("test_cutoffs") or [])]
    if not cutoffs:
        raise ValueError(f"Champion '{predictor.bundle_name}' does not record test_cutoffs; cannot build a drift reference")
    target = get_target(predictor.target)
    frames = []
    for cutoff in cutoffs:
        feats = builder.features_at(cutoff)
        frames.append(feats[target.eligible(feats)])
    reference = pd.concat(frames, ignore_index=True)
    return DriftReference(features=reference, scores=predictor.predict_proba(reference), cutoffs=cutoffs)
