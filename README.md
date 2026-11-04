# Customer Purchase Propensity

CRISP-DM research foundation for a 7-day Retail outcome among previously active users of T-ECD Full. The event meaning and timestamp mapping are not yet verified. See [the project plan](PROJECT_MASTER_PLAN.md), [decisions](docs/decisions.md), and [milestones](MILESTONE_CHECKLIST.md).

## Setup (Windows PowerShell)

Use Python 3.12. After approving installation of dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m tecd_propensity
.\.venv\Scripts\python.exe -m ruff check .
```

The verified Windows Python 3.12 dependency snapshot is in `requirements-lock.txt`: install it with `python -m pip install -r requirements-lock.txt`, then install this project with `python -m pip install --no-deps --no-build-isolation -e .` after ensuring setuptools is available. The Python launcher may not discover an existing installation; use the full path to its python.exe if needed. The lock is a platform-specific snapshot, not a hash-verified lockfile. Configuration is a draft in `config/base.yaml`; no ingestion or modeling is implemented. Tests require no real data. Do not commit data or credentials. The dataset is attributed to [T-ECD](https://huggingface.co/datasets/t-tech/T-ECD), licensed CC BY-NC-SA 4.0; check its terms before use. Do not use this work as evidence of causal marketing impact.
