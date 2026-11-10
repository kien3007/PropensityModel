"""Validate and load the notebook's prepared data without exposing metadata as features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = {'train': (180, 537), 'validation': (551, 621), 'test': (635, 698)}
METADATA = {'household_key', 'snapshot_day', 'split', 'target_any_basket',
            'label_window_touches_anomaly', 'history_90d_touches_anomaly',
            'history_180d_touches_anomaly'}
FEATURES = ['recency_days', 'observed_tenure_days'] + [
    f'{metric}_{window}d'
    for window in (7, 30, 90, 180)
    for metric in ('baskets', 'sales', 'line_items', 'active_days')
]


def load_prepared_split(directory: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    """Load only the requested partition with an explicit, validated feature allowlist."""
    if split not in SPLITS:
        raise ValueError(f'Unknown split: {split}')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('target') != 'target_any_basket' or manifest.get('horizon_days') != 7:
        raise ValueError('Unexpected target contract')
    features = manifest.get('features')
    if features != FEATURES or set(features) & METADATA:
        raise ValueError('Unexpected or unsafe feature contract')
    frame = pd.read_parquet(directory / f'{split}.parquet')
    required = set(FEATURES) | METADATA
    if set(frame.columns) != required:
        raise ValueError('Prepared data columns differ from the expected contract')
    lo, hi = SPLITS[split]
    if frame.empty or not frame['split'].eq(split).all():
        raise ValueError('Empty partition or inconsistent split labels')
    days = frame['snapshot_day']
    if not days.between(lo, hi).all() or not ((days - 180) % 7 == 0).all():
        raise ValueError('Invalid snapshot day')
    if frame.duplicated(['household_key', 'snapshot_day']).any():
        raise ValueError('Duplicate household-snapshot')
    if not frame['target_any_basket'].isin([0, 1]).all():
        raise ValueError('Invalid target values')
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite model features')
    for window in (90, 180):
        expected = days.map(lambda s: any(s-window < d <= s for d in (278, 643)))
        if not frame[f'history_{window}d_touches_anomaly'].eq(expected).all():
            raise ValueError('Incorrect anomaly history flag')
    expected_label = days.map(lambda s: any(s < d <= s+7 for d in (278, 643)))
    if not frame['label_window_touches_anomaly'].eq(expected_label).all():
        raise ValueError('Incorrect anomaly label flag')
    return frame, list(features)
