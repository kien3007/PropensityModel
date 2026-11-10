import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_snapshot_excludes_future_and_includes_zero_sales_basket():
    notebook = json.loads(Path('notebooks/deployment_gia_lap_propensity.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells']
                  if c['cell_type'] == 'code' and 'def make_snapshot' in ''.join(c['source']))
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'make_snapshot')
    features = ['recency_days', 'observed_tenure_days'] + [
        f'{m}_{w}d' for w in (7, 30, 90, 180)
        for m in ('baskets', 'sales', 'line_items', 'active_days')
    ] + ['cadence_median_gap_180d', 'cadence_gap_std_180d', 'cadence_recency_ratio',
         'cadence_has_gap', 'cadence_has_two_gaps']
    scope = {'np': np, 'pd': pd, 'features': features}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<notebook>', 'exec'), scope)
    tx = pd.DataFrame({'household_key': [1, 1, 1, 2], 'BASKET_ID': [11, 12, 13, 21],
                       'DAY': [1, 10, 11, 11], 'SALES_VALUE': [5., 0., 9., 3.]})
    result = scope['make_snapshot'](tx, 10)
    assert result.household_key.tolist() == [1]
    row = result.iloc[0]
    assert row.baskets_7d == 1
    assert row.sales_7d == 0
    assert row.cadence_median_gap_180d == pytest.approx(9)
    assert row.recency_days == 0
