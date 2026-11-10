import ast
import json
from pathlib import Path

import numpy as np
import pytest


def test_top_k_averages_tied_households_without_using_labels_to_rank():
    notebook = json.loads(Path('notebooks/research/baseline_propensity_recency.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells']
                  if c['cell_type'] == 'code' and 'def expected_top_k' in ''.join(c['source']))
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'expected_top_k')
    scope = {'np': np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<notebook>', 'exec'), scope)
    result = scope['expected_top_k']([1, 0, 1, 0], [.9, .8, .8, .8], .5)
    assert result['k'] == 2
    assert result['ties_at_threshold'] == 3
    assert result['tie_slots'] == 1
    assert result['precision_at_k_expected'] == pytest.approx(2/3)
    assert result['lift_at_k_expected'] == pytest.approx(4/3)
