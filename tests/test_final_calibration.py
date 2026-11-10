import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_cadence_never_uses_days_after_cutoff():
    notebook = json.loads(Path('notebooks/research/calibration_va_danh_gia_poc.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells']
                  if c['cell_type'] == 'code' and 'def add_cadence' in ''.join(c['source']))
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'add_cadence')
    cadence = ['cadence_median_gap_180d', 'cadence_gap_std_180d', 'cadence_recency_ratio',
               'cadence_has_gap', 'cadence_has_two_gaps']
    scope = {'pd': pd, 'np': np, 'cadence_features': cadence}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<notebook>', 'exec'), scope)
    panel = pd.DataFrame({'household_key': [1], 'snapshot_day': [10], 'recency_days': [0]})
    active = pd.DataFrame({'household_key': [1, 1, 1, 1], 'DAY': [1, 8, 10, 11]})
    result = scope['add_cadence'](panel, active).iloc[0]
    assert result.cadence_median_gap_180d == pytest.approx(4.5)
    assert result.cadence_has_two_gaps == 1
    assert result.cadence_recency_ratio == 0
