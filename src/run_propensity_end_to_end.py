"""Execute the narrative notebook and save outputs without nbclient."""
from __future__ import annotations

import base64
import io
import json
import os
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')


def main() -> None:
    path = Path('notebooks/customer_purchase_propensity_end_to_end.ipynb')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    settings = Config()
    settings.HistoryManager.hist_file = ':memory:'
    shell = InteractiveShell.instance(config=settings)
    warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive.*')
    warnings.filterwarnings('ignore', message='Could not find the number of physical cores.*')
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        print(f'Executing cell {index}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(''.join(cell['source']), store_history=False)
        outputs = []
        if captured.stdout:
            outputs.append({'output_type': 'stream', 'name': 'stdout',
                            'text': captured.stdout.splitlines(True)})
        if captured.stderr:
            outputs.append({'output_type': 'stream', 'name': 'stderr',
                            'text': captured.stderr.splitlines(True)})
        for output in captured.outputs:
            outputs.append({'output_type': 'display_data', 'data': output.data,
                            'metadata': output.metadata})
        for figure_number in plt.get_fignums():
            figure = plt.figure(figure_number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            outputs.append({'output_type': 'display_data', 'metadata': {},
                            'data': {'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}})
        plt.close('all')
        cell['execution_count'] = index + 1
        cell['outputs'] = outputs
        if result.error_before_exec or result.error_in_exec:
            error = result.error_before_exec or result.error_in_exec
            traceback.print_exception(type(error), error, error.__traceback__)
            raise RuntimeError(f'Notebook cell {index} failed') from error
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Executed {path}')


if __name__ == '__main__':
    main()
