# Build log

## 2026-11-04 ? M00
Initial inspection: workspace had empty README and pyproject, project instructions, and empty scaffold directories; Git reported no repository. Default Python is 3.10.11; `py -0p` reported no installed Pythons; pytest 9.0.3 is available and Ruff is absent. Python 3.12 and dependency installation remain unverified. Created foundation and unreviewed M01 drafts; no data downloads or training. See validation results below. M00 gate remains open pending Python 3.12 bootstrap, dependency verification, CI execution and remaining operational foundation. Next milestone after the M00 gate: M01 review, not yet started.

Validation on Python 3.10.11: `python -m pytest -q` => 1 passed in 0.06s; `python -m tecd_propensity` with PYTHONPATH=src => tecd-propensity 0.1.0; `python -m compileall -q src tests` => exit 0. CI definition added but not executed. No memory-intensive work or downloads performed.

## M00 follow-up
Python 3.12.3 found at the local Python312 installation despite `py -0p` reporting none. Created `.venv`, installed project and dependencies, and saved a pinned environment snapshot. Sandbox network access initially failed (WinError 10013); installation succeeded with escalated network access. `pytest -q`: 3 passed in 0.11s; `ruff check .`: all checks passed after fixing lint findings; `pip check`: no broken requirements; package smoke printed version 0.1.0. Git initialized; `git check-ignore` confirmed .env, data/example.pq, sample_data/example.csv, .venv/example and .pytest_cache/example ignored. CI workflow exists but has not run remotely. No dataset downloaded or model trained. No data-scale memory/disk usage measured; no data-scale work was performed. Next: obtain hosted CI evidence, then review M01; do not advance automatically.

## Local workflow run
Executed workflow commands on existing Windows Python 3.12.3 venv: pip install -e .[dev] succeeded with network access after sandbox WinError 10013; pytest: 3 passed in 0.09s; ruff: all checks passed. This was not a fresh GitHub-hosted runner. No Git remote or commit exists, so hosted workflow was not triggered. M00 hosted CI evidence remains outstanding.
