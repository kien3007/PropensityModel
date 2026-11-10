import json

import pandas as pd
import pytest

from propensity_v2.prepared_data import FEATURES, load_prepared_split


def test_prepared_contract_blocks_metadata_as_features(tmp_path):
    row = {feature: 0.0 for feature in FEATURES}
    row.update(household_key=1, snapshot_day=271, split='train', target_any_basket=1,
               label_window_touches_anomaly=True, history_90d_touches_anomaly=False,
               history_180d_touches_anomaly=False)
    pd.DataFrame([row]).to_parquet(tmp_path / 'train.parquet', index=False)
    manifest = {'target': 'target_any_basket', 'horizon_days': 7, 'features': FEATURES}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    frame, features = load_prepared_split(tmp_path, 'train')
    assert frame.target_any_basket.iloc[0] == 1
    assert 'label_window_touches_anomaly' not in features
    manifest['features'] = FEATURES + ['label_window_touches_anomaly']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='feature contract'):
        load_prepared_split(tmp_path, 'train')
