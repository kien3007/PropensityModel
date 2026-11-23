# Developer shortcuts. Each target is a thin wrapper around a Python entry point, so the
# same commands work without make (e.g. on Windows): copy the command after the tab.
PY ?= python

.PHONY: install lint test test-ci e2e e2e-synthetic train train-lapse batch reproduce analysis load-test feedback

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	$(PY) -m pytest -q

test-ci:
	$(PY) -m pytest -q -m "not integration"

e2e:
	$(PY) -m scripts.e2e_smoke

e2e-synthetic:
	$(PY) -m scripts.e2e_smoke --data synthetic

train:
	$(PY) -m src.pipelines.train_pipeline --target purchase

train-lapse:
	$(PY) -m src.pipelines.train_pipeline --target lapse

batch:
	$(PY) -m src.pipelines.batch_pipeline --cutoff 711

feedback:
	$(PY) -m src.scheduler.daemon --run-job feedback

reproduce:
	$(PY) -m scripts.reproduce_champion --target purchase && $(PY) -m scripts.reproduce_champion --target lapse

analysis:
	$(PY) -m scripts.target_analysis

load-test:
	$(PY) -m scripts.load_test
