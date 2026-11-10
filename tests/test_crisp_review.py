import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def notebook_function(name):
    notebook = json.loads(Path('notebooks/research/kiem_tra_de_xuat_nang_cap_crisp_dm.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                scope = {'np': np, 'pd': pd}
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), scope)
                return scope[name]
    raise AssertionError(f'{name} not found')


def test_ece_includes_endpoints_and_rejects_invalid_scores():
    evaluate = notebook_function('calibration_errors')
    result, bins = evaluate([0, 1, 1], [0, 1, 1], 2)
    assert result['ece'] == pytest.approx(0)
    assert bins.n.sum() == 3
    with pytest.raises(ValueError):
        evaluate([1], [1.1], 2)


def test_trend_requires_household_history_and_uses_disjoint_windows():
    make = notebook_function('add_trend')
    data = {'observed_tenure_days': [29, 28]}
    for metric in ['baskets', 'sales', 'active_days']:
        data[f'{metric}_90d'] = [9., 9.]
        data[f'{metric}_180d'] = [18., 18.]
    for metric in ['baskets', 'sales']:
        data[f'{metric}_7d'] = [7., 7.]
        data[f'{metric}_30d'] = [30., 30.]
    out, _ = make(pd.DataFrame(data))
    assert out.loc[0, 'baskets_trend_7_vs_previous23'] == pytest.approx(0)
    assert pd.isna(out.loc[1, 'baskets_trend_7_vs_previous23'])
    assert pd.isna(out.loc[0, 'baskets_trend_90_vs_previous90'])


def test_power_uses_the_specified_control_rate():
    from scipy.stats import norm

    notebook = json.loads(Path('notebooks/research/kiem_tra_de_xuat_nang_cap_crisp_dm.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'def required_per_arm' in ''.join(cell['source']))
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'required_per_arm')
    scope = {'np': np, 'za': norm.ppf(.975), 'zb': norm.ppf(.8)}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<notebook>', 'exec'), scope)
    required = scope['required_per_arm']
    assert np.ceil(required(.5322, .58542)) == 1366
    assert np.ceil(required(.15, .165)) == 9257
