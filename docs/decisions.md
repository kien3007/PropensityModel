# Decision log

## 2026-11-04 ? M00 foundation
- Use a src-layout package and YAML configuration; config values are proposals, not verified data facts. Seed 42 and 0.5% are initial reproducibility defaults, not evidence of sample adequacy.
- Dependencies are declared but unpinned until verified on Python 3.12. No installs authorized or attempted.
- M01 documents are drafts only; semantic definitions and stakeholder review remain open. Source schema, timestamp and outcome semantics await M03.
- Python 3.12 environment, CI execution, operational run manifest/logging/checksum utilities, and Git initialization remain unverified or incomplete; M00 gate is open. No source data was downloaded.

## 2026-11-04 ? M00 validation follow-up
- Python 3.12.3 was found outside launcher discovery; the local venv and declared dependencies installed successfully. `requirements-lock.txt` records the resolved Windows environment, without artifact hashes.
- Git was initialized; ignore rules were checked for representative secrets, data, cache and venv paths. No commit was created.
- Runtime utilities provide bounded checksum, free disk query, unique run ID, JSON event logging and exclusive manifest write with dry-run. They are foundation utilities, not an ingestion pipeline.
- Local Python 3.12 tests and lint pass. Hosted CI has not run because no remote or commit exists; gate evidence for hosted CI remains outstanding. Source semantics and M01 review remain unresolved.
